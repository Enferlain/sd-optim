import React, { useState } from 'react';
import MainConfigTab from './MainConfigTab.js';
import OptimizationGuideTab from './OptimizationGuideTab.js';
import CargoTab from './CargoTab.js';
import CustomSelect from './CustomSelect.js'; // <-- Import our new component!
import styles from './ConfigurationManager.module.css';

function ConfigurationManager() {
  const [activeTab, setActiveTab] = useState('Main Config');
  const tabOptions = ['Main Config', 'Optimization Guide', 'Cargo'];

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
    <div className={styles.layoutContainer}>
      <div className={styles.sidebar}>
        {/* Replace the old <select> with our new component! */}
        <CustomSelect
            options={tabOptions}
            value={activeTab}
            onChange={(newTab) => setActiveTab(newTab)}
        />
      </div>

      <div className={styles.contentPanel}>
        {renderTabContent()}
      </div>
    </div>
  );
}

export default ConfigurationManager;
