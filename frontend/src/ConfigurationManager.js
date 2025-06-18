import React, { useState } from 'react';
import MainConfigTab from './MainConfigTab.js';
import OptimizationGuideTab from './OptimizationGuideTab.js';
import PayloadsTab from './PayloadsTab.js';
import CustomSelect from './CustomSelect.js';
import styles from './ConfigurationManager.module.css';

// Placeholder for the glyph card, so the selection panel has something to show
const GlyphCard = ({ name }) => (
  <div className={styles.glyphCard}>
    <div className={styles.glyphVisual}>◆</div>
    <p className={styles.glyphName}>{name}</p>
  </div>
);

function ConfigurationManager() {
  const [activeTab, setActiveTab] = useState('Payload Workshop');
  const tabOptions = ['Main Config', 'Payload Workshop', 'Optimization Guide'];
  const [selectedPayloads, setSelectedPayloads] = useState(['payload_three']);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'Main Config':
        return <MainConfigTab />;
      case 'Payload Workshop':
        return <PayloadsTab />;
      case 'Optimization Guide':
        return <OptimizationGuideTab />;
      default:
        return <div>Select a tab</div>;
    }
  };

  return (
    // This is now our GRID container. Its children are the grid items.
    <div className={styles.pageGrid}>
      
      {/* Grid Item 1: The Tab Selector */}
      <div className={styles.sidebarContainer}>
        <CustomSelect
            options={tabOptions}
            value={activeTab}
            onChange={(newTab) => setActiveTab(newTab)}
        />
      </div>

      {/* Grid Item 2: The Selection Panel (conditional) */}
      {activeTab === 'Payload Workshop' && (
        <div className={styles.selectionContainer}>
            <h3>Selected Payloads</h3>
            <div className={styles.selectionBox}>
                {/* ... content ... */}
            </div>
        </div>
      )}

      {/* Grid Item 3: The Main Content Panel */}
      <div className={styles.contentContainer}>
        {renderTabContent()}
      </div>
    </div>
  );
}

export default ConfigurationManager;