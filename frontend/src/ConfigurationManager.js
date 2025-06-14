import React, { useState } from 'react';
import MainConfigTab from '/home/user/sdoptimui/sd-optim/frontend/src/MainConfigTab.js';
import OptimizationGuideTab from '/home/user/sdoptimui/sd-optim/frontend/src/OptimizationGuideTab.js';
import CargoTab from '/home/user/sdoptimui/sd-optim/frontend/src/CargoTab.js';
import styles from '/home/user/sdoptimui/sd-optim/frontend/src/ConfigurationManager.module.css';

function ConfigurationManager() {
  const [activeTab, setActiveTab] = useState('Main Config'); // State to track active tab

  const renderTabContent = () => {
    // Remove any previous state setting before rendering new tab
    // (Optional, depending on desired behavior)
    // setActiveTab(activeTab); 

    switch (activeTab) {
      case 'Main Config':
        return <MainConfigTab />;
      case 'Optimization Guide':
        return <OptimizationGuideTab />;
      case 'Cargo':
        return <CargoTab />;
      default:
        return <div>Select a tab</div>;
    }
  };

  return (
    <div className={styles.container}>
      <select onChange={(e) => setActiveTab(e.target.value)} value={activeTab}>
        <option value="Main Config">Main Config</option>
        <option value="Optimization Guide">Optimization Guide</option>
        <option value="Cargo">Cargo</option>
      </select>


      <div style={{ marginTop: '20px' }}>
        {renderTabContent()}
      </div>
    </div>
  );
}

export default ConfigurationManager;