import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append('image', selectedImage);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow p-6">
        <h1 className="text-2xl font-bold text-center mb-6">Marine Animal Classifier</h1>
        
        <div className="space-y-6">
          <div className="flex justify-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="cursor-pointer bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Select Image
            </label>
          </div>

          {preview && (
            <div className="mt-4">
              <div className="relative w-full h-64">
                <img
                  src={prediction?.annotated_image ? `data:image/jpeg;base64,${prediction.annotated_image}` : preview}
                  alt="Preview"
                  className="w-full h-full object-contain"
                />
              </div>
              
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="mt-4 w-full bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:bg-gray-400"
              >
                {loading ? 'Processing...' : 'Classify Image'}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 text-red-500 text-center">
              {error}
            </div>
          )}

          {prediction && !error && (
            <div className="mt-4 text-center">
              <h3 className="text-xl font-semibold">
                Prediction: {prediction.class}
              </h3>
              <p className="text-gray-600">
                Confidence: {(prediction.confidence * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;