import React, { useState } from 'react';

const LandingPage = ({ onBackendLaunch }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleLaunch = () => {
    setIsLoading(true);
    // Simulate backend launch for UI development
    setTimeout(() => {
        onBackendLaunch();
        setIsLoading(false);
    }, 500);
  };

  return (
    // We use the same structure as your inspiration's entry overlay!
    <div id="enter-overlay">
        <div className="enter-content">
            <h1>sd-optim</h1>
            <p>Click to enter the configuration interface.</p>
            <button 
                id="enter-btn" 
                className="btn btn--primary" // Using classes from style.css
                onClick={handleLaunch} 
                disabled={isLoading}
            >
                {isLoading ? 'Launching...' : 'Enter'}
            </button>
        </div>
    </div>
  );
};

export default LandingPage;