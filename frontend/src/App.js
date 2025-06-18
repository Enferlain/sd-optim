import React, { useState } from 'react';
import LandingPage from './LandingPage.js';
import Workbench from './Workbench.js'; // Import our new main component
import './App.css'; 

function App() {
  const [isBackendLaunched, setIsBackendLaunched] = useState(false);

  const handleBackendLaunch = () => {
    setIsBackendLaunched(true);
  };

  return (
    <div>
      {/* We now launch the Workbench instead of the old ConfigurationManager */}
      {isBackendLaunched ? <Workbench /> : <LandingPage onBackendLaunch={handleBackendLaunch} />}
    </div>
  );
}

export default App;