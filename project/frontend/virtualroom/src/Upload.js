import React, { useRef, useState, useEffect } from "react";
import "./App.css";

export default function Upload() {
  const fileInputRef1 = useRef(null);
  const fileInputRef2 = useRef(null);
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [generationStatus, setGenerationStatus] = useState("");
  const [uploadedImageUrls, setUploadedImageUrls] = useState({ dress: null, person: null });
  const [isUploaded, setIsUploaded] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [statusCheckInterval, setStatusCheckInterval] = useState(null);
  const [resultImageUrl, setResultImageUrl] = useState(null);

  // Function to check job status
  const checkJobStatus = (id) => {
    fetch(`http://127.0.0.1:5001/status/${id}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Status check failed");
        }
        return response.json();
      })
      .then((data) => {
        setProcessingStatus(data);
        
        // If result URL is available, set it
        if (data.result_url) {
          setResultImageUrl(data.result_url);
        }
        
        // If processing is complete or failed, stop checking
        if (
          data.overall_status === "completed" ||
          data.overall_status === "failed"
        ) {
          clearInterval(statusCheckInterval);
          setStatusCheckInterval(null);
        }
      })
      .catch((error) => {
        console.error("Error checking status:", error);
      });
  };

  // Clean up old datasets when a result is displayed
  useEffect(() => {
    if (resultImageUrl && jobId) {
      // Send a request to clean up old datasets
      fetch("http://127.0.0.1:5001/cleanup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          job_id: jobId,
        }),
      }).catch((error) => {
        console.error("Error cleaning up datasets:", error);
      });
    }
  }, [resultImageUrl, jobId]);

  // Clear interval when component unmounts
  useEffect(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
      }
    };
  }, [statusCheckInterval]);

  const handleFileChange1 = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage1(file);
    }
  };

  const handleFileChange2 = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage2(file);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!image1 || !image2) {
      setUploadStatus("Please select both images.");
      return;
    }

    const formData = new FormData();
    formData.append("dress_image", image1);
    formData.append("person_image", image2);

    setUploadStatus("Uploading...");
    
    fetch("http://127.0.0.1:5001/upload-images", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Upload failed");
        }
        return response.json();
      })
      .then((data) => {
        setUploadStatus("Upload successful!");
        setUploadedImageUrls({
          dress: data.dress_image_url,
          person: data.person_image_url
        });
        setIsUploaded(true);
        console.log("Success:", data);
      })
      .catch((error) => {
        setUploadStatus("Error uploading images");
        console.error("Error:", error);
      });
  };

  const handleGenerate = () => {
    if (!uploadedImageUrls.dress || !uploadedImageUrls.person) {
      setGenerationStatus("Please upload images first.");
      return;
    }

    setGenerationStatus("Sending request to generate...");
    // Reset result image if making a new request
    setResultImageUrl(null);

    fetch("http://127.0.0.1:5001/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dress_image_path: uploadedImageUrls.dress,
        person_image_path: uploadedImageUrls.person,
      }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Generation request failed");
        }
        return response.json();
      })
      .then((data) => {
        setJobId(data.job_id);
        setGenerationStatus(`Generation request submitted successfully! Job ID: ${data.job_id}`);
        console.log("Generation Success:", data);
        
        // Start checking status every 5 seconds
        const intervalId = setInterval(() => checkJobStatus(data.job_id), 5000);
        setStatusCheckInterval(intervalId);
        
        // Initial status check
        checkJobStatus(data.job_id);
      })
      .catch((error) => {
        setGenerationStatus("Error submitting generation request");
        console.error("Error:", error);
      });
  };

  // Helper function to render status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case "completed":
        return "✅";
      case "processing":
        return "⏳";
      case "failed":
        return "❌";
      case "pending":
      default:
        return "⏳";
    }
  };

  return (
    <div className="upload-container">
      <h2>Virtual Dressing Room</h2>
      
      <div className="main-content">
        <div className="upload-section">
          <form onSubmit={handleSubmit}>
            <div className="image-flex-container">
              {/* Image Box 1 */}
              <div className="image-box">
                <input
                  type="file"
                  ref={fileInputRef1}
                  accept="image/*"
                  onChange={handleFileChange1}
                  className="file-input"
                />
                <div className="preview-container">
                  {image1 ? (
                    <img
                      src={URL.createObjectURL(image1)}
                      alt="Dress Preview"
                      className="preview-image"
                    />
                  ) : (
                    <p className="placeholder-text">Upload Dress Image</p>
                  )}
                </div>
                {image1 && <p className="file-name">{image1.name}</p>}
                <button
                  type="button"
                  onClick={() => fileInputRef1.current.click()}
                  className="choose-button"
                >
                  Choose Dress Image
                </button>
              </div>

              {/* Image Box 2 */}
              <div className="image-box">
                <input
                  type="file"
                  ref={fileInputRef2}
                  accept="image/*"
                  onChange={handleFileChange2}
                  className="file-input"
                />
                <div className="preview-container">
                  {image2 ? (
                    <img
                      src={URL.createObjectURL(image2)}
                      alt="Person Preview"
                      className="preview-image"
                    />
                  ) : (
                    <p className="placeholder-text">Upload Person Image</p>
                  )}
                </div>
                {image2 && <p className="file-name">{image2.name}</p>}
                <button
                  type="button"
                  onClick={() => fileInputRef2.current.click()}
                  className="choose-button"
                >
                  Choose Person Image
                </button>
              </div>
            </div>
            <button type="submit" className="upload-button">
              Upload Images
            </button>
          </form>
          {uploadStatus && <p className="status-message">{uploadStatus}</p>}

          {isUploaded && (
            <div className="generate-section">
              <button 
                onClick={handleGenerate} 
                className="generate-button"
                disabled={!!jobId && processingStatus?.overall_status === "processing"}
              >
                Generate Virtual Try-On
              </button>
              {generationStatus && <p className="status-message">{generationStatus}</p>}
            </div>
          )}
        </div>
        
        {/* Result Display Section */}
        {resultImageUrl && (
          <div className="result-section">
            <h3>Virtual Try-On Result</h3>
            <div className="result-container">
              <img src={`http://127.0.0.1:5001/result-image/${jobId}`} alt="Virtual Try-On Result" className="result-image" />
            </div>
            <a 
              href={`http://127.0.0.1:5001/result-image/${jobId}`}
              download="virtual-try-on-result.jpg"
              className="download-button"
              target="_blank"
              rel="noopener noreferrer"
            >
              Download Result
            </a>
          </div>
        )}
      </div>
      
      {/* Processing Status Section */}
      {processingStatus && (
        <div className="processing-status">
          <h3>Processing Status</h3>
          <div className="status-details">
            <div className="status-item">
              <span className="status-label">Overall Status:</span>
              <span className="status-value">
                {getStatusIcon(processingStatus.overall_status)} {processingStatus.overall_status}
              </span>
            </div>
            <h4>Preprocessing Steps:</h4>
            <div className="status-item">
              <span className="status-label">Remove Background:</span>
              <span className="status-value">
                {getStatusIcon(processingStatus.preprocessing.remove_bg)} {processingStatus.preprocessing.remove_bg}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Cloth Mask Creation:</span>
              <span className="status-value">
                {getStatusIcon(processingStatus.preprocessing.cloth_mask)} {processingStatus.preprocessing.cloth_mask}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Segmentation:</span>
              <span className="status-value">
                {getStatusIcon(processingStatus.preprocessing.segmentation)} {processingStatus.preprocessing.segmentation}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Pose Generation:</span>
              <span className="status-value">
                {getStatusIcon(processingStatus.preprocessing.pose_generation)} {processingStatus.preprocessing.pose_generation}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Cloth Resize:</span>
              <span className="status-value">
                {getStatusIcon(processingStatus.preprocessing.cloth_resize)} {processingStatus.preprocessing.cloth_resize}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}