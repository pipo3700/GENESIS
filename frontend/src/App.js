import React, { useState } from 'react';
import logoRight from './assets/images/upv.png';
import logoLeft from './assets/images/etsinf.png';
import './App.css';

function App() {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!file) {
      alert("Please select a document.");
      return;
    }

    const formData = new FormData();
    formData.append("cv", file);

    // Aquí iría tu lógica de envío a Azure Function
    console.log("Document selected:", file);
    alert("Simulating sending to backend..");
  };

  return (
    <div className="App">
      <h1>CV Generator</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="jobOffer">Introduce the job offer:</label>
          <textarea id="jobOffer" rows="5" placeholder="Describe the job offer..." />
        </div>

        <div>
          <label htmlFor="cv">Select your CV:</label>
          <label className="custom-file-upload">
            <input type="file" id="cv" accept=".pdf,.doc,.docx,.txt" onChange={handleFileChange} />
            Upload file
          </label>
          {file && <p>Selected: {file.name}</p>}
        </div>

        <button type="submit">Generate your ideal CV</button>
      </form>
      <img src={logoLeft} alt="ETSINF logo" className="bottom-left" />
      <img src={logoRight} alt="UPV logo" className="bottom-right" />
    </div>
  );
}

export default App;
