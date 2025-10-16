import React, { useState, useEffect } from 'react';

function App() {
  const [data, setData] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/api/data')
      .then(response => response.json())
      .then(data => setData(data.data))
      .catch(error => console.error('Error:', error));
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>React + FastAPI App</h1>
      <p>Data from backend: {data}</p>
    </div>
  );
}

export default App;
