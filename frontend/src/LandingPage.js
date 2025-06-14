import React, { useState } from 'react';
// Assume runTerminalCommand is available globally or imported
// import { runTerminalCommand } from '../utils/terminal'; 

const LandingPage = ({ onBackendLaunch }) => {
  const [isLoading, setIsLoading] = useState(false);
  const pageStyles = {
    width: '100vw',
    height: '100vh',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    // Placeholder for blurred background
    backgroundColor: '#cccccc', // Example placeholder color
    filter: 'blur(0px)', // Placeholder blur
    backgroundImage: 'url(\'\')', // Placeholder for background image
    backgroundSize: 'cover',
    backgroundPosition: 'center',
  };

  const buttonStyles = {
    padding: '10px 20px',
    fontSize: '1.2em',
    cursor: 'pointer',
  };

  const handleLaunch = async () => {
    setIsLoading(true);
    try {
      // Comment out or remove the actual backend launch command for now
      // console.log("Launching backend script...");
      // // Assuming runTerminalCommand is available
      // await runTerminalCommand("python sd-optim/sd_optim.py");
      // console.log("Backend script launched.");

      // Only call the prop to trigger UI transition
      console.log("Backend script launched.");
      onBackendLaunch(); // Call the prop to signal backend launch
      // You might want to navigate to the next page here
    } catch (error) {
      console.error("Error launching backend script:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={pageStyles}>
      <button style={buttonStyles} onClick={handleLaunch} disabled={isLoading}>{isLoading ? 'Launching...' : 'Enter/Launch'}</button>
    </div>
  );
};

export default LandingPage;