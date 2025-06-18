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
  const [activeTab, setActiveTab] = useState('Payload Workshop'); // Start here to see the new layout
  const tabOptions = ['Main Config', 'Payload Workshop', 'Optimization Guide'];

  // This state will be lifted up here later!
  const [selectedPayloads, setSelectedPayloads] = useState(['payload_three']);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'Main Config':
        return <MainConfigTab />;
      case 'Payload Workshop':
        // We will pass down the state and functions to modify it later
        return <PayloadsTab />; 
      case 'Optimization Guide':
        return <OptimizationGuideTab />;
      default:
        return <div>Select a tab</div>;
    }
  };

  return (
    // This is now the main container for the entire three-panel layout
    <div className={styles.pageLayout}>
      {/* Panel 1: The Primary Sidebar with the tab selector */}
      <div className={styles.primarySidebar}>
        <CustomSelect
            options={tabOptions}
            value={activeTab}
            onChange={(newTab) => setActiveTab(newTab)}
        />
      </div>

      {/* Panel 2: The Selection Panel (conditionally rendered!) */}
      {activeTab === 'Payload Workshop' && (
        <div className={styles.selectionPanel}>
            <h3>Selected Payloads</h3>
            <div className={styles.selectionBox}>
                {selectedPayloads.map(name => <GlyphCard key={name} name={name} />)}
            </div>
        </div>
      )}

      {/* Panel 3: The Main Content Panel */}
      <div className={styles.contentPanel}>
        {renderTabContent()}
      </div>
    </div>
  );
}

export default ConfigurationManager;