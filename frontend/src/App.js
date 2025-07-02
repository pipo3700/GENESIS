import React, { useState } from 'react';
import logoRight from './assets/images/upv.png';
import logoLeft from './assets/images/etsinf.png';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.size > 5 * 1024 * 1024) {
      alert("El archivo es demasiado grande. Máximo 5MB permitido.");
      e.target.value = '';
      return;
    }
    setFile(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Please select a document.");
    const jobOffer = document.getElementById("jobOffer").value;
    if (!jobOffer.trim()) return alert("Please describe your job offer.");
    setIsUploading(true);

    const formData = new FormData();
    formData.append("cv", file);
    formData.append("jobOffer", jobOffer);

    try {
      const response = await fetch("https://api-genesis.azure-api.net/func-genesis-upload/UploadCVandOffer", {
        method: "POST",
        body: formData,
        headers: {
          "Ocp-Apim-Subscription-Key": "75efa2c8485243f5ae8397740e084a8e"
        }
      });

      const data = await response.json();
      if (response.ok) {
        alert("¡Subida exitosa! 🎉 Puedes generar tu CV adaptado.");
        setJobId(data.jobId);
        setFile(null);
        document.getElementById("jobOffer").value = "";
        document.getElementById("cv").value = "";
      } else {
        alert("Error: " + (data.message || "Upload failed"));
      }
    } catch (err) {
      alert("Network error: " + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleGenerate = async () => {
    if (!jobId) return alert("No se ha subido ningún archivo aún.");
    setIsGenerating(true);
    try {
      const response = await fetch("https://api-genesis.azure-api.net/func-genesis-generate/GenerateAdaptedCV", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Ocp-Apim-Subscription-Key": "75efa2c8485243f5ae8397740e084a8e"
        },
        body: JSON.stringify({ jobId })
      });

      let result;
      try {
        result = await response.json();
      } catch (e) {
        const text = await response.text();
        console.error("Respuesta no JSON:", text);
        alert("Error inesperado en el servidor.");
        return;
      }

      if (response.ok && result.generatedCvUrl) {
        // Descargar automáticamente
        const a = document.createElement("a");
        a.href = result.generatedCvUrl;
        a.download = `CV_Adaptado_${jobId}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      } else {
        alert("Error: " + result.message);
      }
    } catch (err) {
      alert("Error generando el CV: " + err.message);
    } finally {
      setIsGenerating(false);
    }
  };

const handleGenerateFineTuned = async () => {
  if (!jobId) return alert("No se ha subido ningún archivo aún.");
  setIsGenerating(true);
  try {
    const response = await fetch("https://api-genesis.azure-api.net/func-genesis-generate-phase2/GenerateCVadaptedphase2", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": "75efa2c8485243f5ae8397740e084a8e"
      },
      body: JSON.stringify({ jobId })
    });

    const raw = await response.text();
    let result;
    try {
      result = JSON.parse(raw);
    } catch (e) {
      console.error("Respuesta no es JSON:", raw);
      alert("Error inesperado en el servidor.");
      return;
    }

    if (response.ok && result.generatedCvUrl) {
      const a = document.createElement("a");
      a.href = result.generatedCvUrl;
      a.download = `CV_Adaptado_FT_${jobId}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } else {
      alert("Error: " + (result.message || "No se pudo generar el CV"));
    }

  } catch (err) {
    alert("Error generando el CV fine-tuned: " + err.message);
  } finally {
    setIsGenerating(false);
  }
};



  return (
    <div className="App">
      <h1>Adapted CV Generator</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="jobOffer">Introduce your Job Offer:</label>
          <textarea id="jobOffer" rows="5" placeholder="Describe the job offer..." disabled={isUploading} />
        </div>

        <div>
          <label htmlFor="cv">Select your CV:</label>
          <label className="custom-file-upload">
            <input type="file" id="cv" accept=".pdf,.doc,.docx,.txt" onChange={handleFileChange} disabled={isUploading} />
            {isUploading ? 'Uploading...' : 'Upload file'}
          </label>
          {file && <p>Seleccionado: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</p>}
        </div>

        <button type="submit" disabled={isUploading || !file}>
          {isUploading ? 'Uploading...' : 'Upload my CV and job Offer'}
        </button>
      </form>

      {jobId && (
        <div style={{ marginTop: '20px' }}>
          <button onClick={handleGenerate} disabled={isGenerating}>
            {isGenerating ? 'Generating CV...' : 'Generar CV Adaptado'}
          </button>

          <button
            onClick={handleGenerateFineTuned}
            disabled={isGenerating}
            style={{ marginLeft: '10px', backgroundColor: '#4CAF50', color: 'white' }}
          >
            {isGenerating ? 'Generating CV...' : 'Generar CV Adaptado (Fine-tuning)'}
          </button>
        </div>
      )}


      <img src={logoLeft} alt="ETSINF logo" className="bottom-left" />
      <img src={logoRight} alt="UPV logo" className="bottom-right" />
    </div>
  );
}

export default App;
