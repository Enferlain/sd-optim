import React, { useState, useEffect } from 'react';

const MainConfigTab = () => {
  const [config, setConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/config')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setConfig(data);
        setIsLoading(false);
      })
      .catch(error => {
        setError(error);
        setIsLoading(false);
      });
  }, []);

  return (
    <div>
      <h2>Main Configuration</h2>
      {isLoading && <p>Loading configuration...</p>}
      {error && <p>Error loading configuration: {error.message}</p>}
      {config && !isLoading && !error && <p>Config loaded. Build form here.</p>}
    </div>
  );
};

export default MainConfigTab;