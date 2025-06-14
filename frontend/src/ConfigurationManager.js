import React, { useState } from 'react';
import MainConfigTab from './MainConfigTab.js';
import OptimizationGuideTab from './OptimizationGuideTab.js';
import CargoTab from './CargoTab.js';
import styles from './ConfigurationManager.module.css';

function ConfigurationManager() {
  const [activeTab, setActiveTab] = useState('Main Config');

  const renderTabContent = () => {
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
    // --- THIS IS THE NEW LAYOUT STRUCTURE ---
    <div className={styles.layoutContainer}>
      {/* The sidebar is now its own separate element */}
      <div className={styles.sidebar}>
        <select onChange={(e) => setActiveTab(e.target.value)} value={activeTab} className={styles.tabSelect}>
          <option>Main Config</option>
          <option>Optimization Guide</option>
          <option>Cargo</option>
        </select>
        {/* We can add other things to the sidebar later! */}
      </div>

      {/* The main content panel is also its own element */}
      <div className={styles.contentPanel}>
        {renderTabContent()}
      </div>
    </div>
  );
}

export default ConfigurationManager;