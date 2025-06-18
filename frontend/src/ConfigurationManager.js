import React, { useState } from 'react';
import MainConfigTab from './MainConfigTab.js';
import OptimizationGuideTab from './OptimizationGuideTab.js';
import PayloadsTab from './PayloadsTab.js'; // <-- Changed from CargoTab to PayloadsTab
import CustomSelect from './CustomSelect.js';
import styles from './ConfigurationManager.module.css';

function ConfigurationManager() {
  const [activeTab, setActiveTab] = useState('Main Config');
  // Let's update the options here too!
  const tabOptions = ['Main Config', 'Payload Workshop', 'Optimization Guide'];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'Main Config':
        return <MainConfigTab />;
      case 'Payload Workshop': // <-- And here
        return <PayloadsTab />;
      case 'Optimization Guide':
        return <OptimizationGuideTab />;
      default:
        return <div>Select a tab</div>;
    }
  };

  return (
    <div className={styles.layoutContainer}>
      <div className={styles.sidebar}>
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