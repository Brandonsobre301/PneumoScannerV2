
import React, { useState, useRef } from "react";
import "./App.css";

export interface PredictionResponse {
  status: "success" | "ood";
  prediction: "NORMAL" | "PNEUMONIA" | null;
  confidence: number | null;
  message?: string | null;
  ood_metrics: {
    cosine_distance: number | null;
    mc_variance: number | null;
    gatekeeper_confidence?: number | null;
  };
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setupFile(e.target.files[0]);
    }
  };

  const setupFile = (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setupFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleClickDropzone = () => {
    fileInputRef.current?.click();
  };

  const loadSample = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type });
      setupFile(file);
    } catch (err) {
      console.error("Error loading sample", err);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to analyze image");
      }

      const data: PredictionResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearSelection = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>PneumoScannerV2</h1>
        <p>Automated Pneumonia Detection via Chest X-ray</p>
      </header>

      <main className="main-content">
        <div
          className="dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={handleClickDropzone}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="file-input"
          />

          {!previewUrl ? (
            <div className="dropzone-content">
              <svg className="upload-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="var(--accent)"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
              <p>Drag & drop an X-ray image here, or browse.</p>
            </div>
          ) : (
            <div className="preview-container">
              <img src={previewUrl} alt="X-ray preview" className="preview-image" />
              {isAnalyzing && <div className="scanner-overlay"></div>}
            </div>
          )}
        </div>

        {!previewUrl && (
          <div className="samples-section">
            <h3>Try a sample X-ray:</h3>
            <div className="sample-images">
              <button 
                className="sample-btn"
                onClick={() => loadSample("https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=800&q=80", "sample_normal.jpg")}
              >Sample 1</button>
              <button 
                className="sample-btn"
                onClick={() => loadSample("https://images.unsplash.com/photo-1559703248-dcaaec9fab78?auto=format&fit=crop&w=800&q=80", "sample_pneumonia.jpg")}
              >Sample 2</button>
            </div>
          </div>
        )}

        <div className="actions">
          {previewUrl && (
             <button className="btn btn-secondary" onClick={clearSelection} disabled={isAnalyzing}>
               Clear
             </button>
          )}
          <button
            className="btn btn-primary"
            onClick={analyzeImage}
            disabled={!previewUrl || isAnalyzing}
          >
            {isAnalyzing ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && result.status === "ood" && (
          <div className="results-panel ood-warning">
            <h3>Out-of-Distribution Warning</h3>
            <p className="ood-message">
              {result.message || "Invalid Image Detected: High uncertainty or out-of-distribution input."}
            </p>
          </div>
        )}

        {result && result.status === "success" && result.prediction && result.confidence !== null && (
          <div className="results-panel">
            <h3>Analysis Complete</h3>
            <div className="result-item">
              <span className="result-label">Prediction:</span>
              <span className={`result-value ${result.prediction.toLowerCase()}`}>
                {result.prediction.toUpperCase()}
              </span>
            </div>
            <div className="result-item">
              <span className="result-label">Confidence:</span>
              <span className="result-value">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

