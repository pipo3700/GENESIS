import React, { useState } from 'react';
import logoRight from './assets/images/upv.png';
import logoLeft from './assets/images/etsinf.png';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    // Validar tamaño del archivo (máximo 5MB para Lambda)
    if (selectedFile && selectedFile.size > 5 * 1024 * 1024) {
      alert("El archivo es demasiado grande. Máximo 5MB permitido.");
      e.target.value = ''; // Limpiar input
      return;
    }
    
    setFile(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert("Please select a document.");
      return;
    }

    const jobOffer = document.getElementById("jobOffer").value;
    if (!jobOffer.trim()) {
      alert("Please describe your job offer.");
      return;
    }

    setIsUploading(true);

    const formData = new FormData();
    formData.append("cv", file);
    formData.append("jobOffer", jobOffer);

    // Log para debugging
    console.log("Uploading file:", file.name, "Size:", file.size, "Type:", file.type);
    console.log("Job offer length:", jobOffer.length);

    try {
      const response = await fetch("https://apitfmgenesis.azure-api.net/upload/UploadCVandOffer", {
        method: "POST",
        body: formData,
        headers: {
          "Ocp-Apim-Subscription-Key": "5c00f140db924048b40712e65ed7f928"
        }
      });

      console.log("Response status:", response.status);
      console.log("Response headers:", Object.fromEntries(response.headers.entries()));

      let data;
      try {
        data = await response.json();
      } catch (jsonError) {
        console.error("Error parsing JSON response:", jsonError);
        const textResponse = await response.text();
        console.log("Raw response:", textResponse);
        throw new Error("Respuesta del servidor no es JSON válido");
      }

      if (response.ok) {
        alert("¡Subida exitosa! 🎉");
        console.log("Server response:", data);
        
        // Limpiar formulario
        setFile(null);
        document.getElementById("jobOffer").value = "";
        document.getElementById("cv").value = "";
      } else {
        alert("Error: " + (data.message || "Upload failed"));
        console.error("Error response:", data);
      }
    } catch (err) {
      console.error("Network error:", err);
      alert("Network error while the file is uploading: " + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="App">
      <h1>CV Generator</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="jobOffer">Introduce your Job Offer:</label>
          <textarea 
            id="jobOffer" 
            rows="5" 
            placeholder="Describe the job offer..." 
            disabled={isUploading}
          />
        </div>

        <div>
          <label htmlFor="cv">Select your CV:</label>
          <label className="custom-file-upload">
            <input 
              type="file" 
              id="cv" 
              accept=".pdf,.doc,.docx,.txt" 
              onChange={handleFileChange}
              disabled={isUploading}
            />
            {isUploading ? 'Uploading...' : 'Upload file'}
          </label>
          {file && (
            <p>
              Seleccionado: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        <button type="submit" disabled={isUploading || !file}>
          {isUploading ? 'Generating...' : 'Generate your ideal CV'}
        </button>
      </form>
      
      {isUploading && (
        <div style={{marginTop: '20px', textAlign: 'center'}}>
          <p>Uploading file, please wait...</p>
        </div>
      )}
      
      <img src={logoLeft} alt="ETSINF logo" className="bottom-left" />
      <img src={logoRight} alt="UPV logo" className="bottom-right" />
    </div>
  );
}

export default App;