import React, { useState } from 'react';
import LandingPage from '/home/user/sdoptimui/sd-optim/frontend/src/LandingPage.js';
import ConfigurationManager from '/home/user/sdoptimui/sd-optim/frontend/src/ConfigurationManager.js'; // Import the new component
import './App.css'; // <-- ADD THIS IMPORT!

function App() {
  const [isBackendLaunched, setIsBackendLaunched] = useState(false);

  const handleBackendLaunch = () => {
    setIsBackendLaunched(true);
  };

  return (
    <div>
      {isBackendLaunched ? <ConfigurationManager /> : <LandingPage onBackendLaunch={handleBackendLaunch} />}
    </div>
  );
}

export default App;