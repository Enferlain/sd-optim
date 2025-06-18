import React, { useState } from 'react';
import LandingPage from './LandingPage.js';
import Workbench from './Workbench.js'; // <-- Make sure it says Workbench here!
import './App.css'; 

function App() {
  const [isBackendLaunched, setIsBackendLaunched] = useState(false);

  const handleBackendLaunch = () => {
    setIsBackendLaunched(true);
  };

  return (
    <div>
      {isBackendLaunched ? <Workbench /> : <LandingPage onBackendLaunch={handleBackendLaunch} />}
    </div>
  );
}

export default App;