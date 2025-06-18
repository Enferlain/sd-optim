import React, { useState, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import styles from './Workbench.module.css';

import CustomSelect from './CustomSelect.js'; // We need our dropdown back!
import MainConfigTab from './MainConfigTab.js';
import PayloadsTab from './PayloadsTab.js';

// --- Reusable Window Component (No changes) ---
const Window = ({ panel, onDragStop, onResizeStop, onBringToFront, children }) => {
  return (
    <Rnd
      style={{ zIndex: panel.zIndex }}
      className={styles.glassPanel}
      size={{ width: panel.width, height: panel.height }}
      position={{ x: panel.x, y: panel.y }}
      minWidth={250}
      minHeight={200}
      onDragStart={() => onBringToFront(panel.id)}
      onDragStop={(e, d) => onDragStop(panel.id, { x: d.x, y: d.y })}
      onResizeStart={() => onBringToFront(panel.id)}
      onResizeStop={(e, direction, ref, delta, position) => {
        onResizeStop(panel.id, { width: ref.style.width, height: ref.style.height }, position);
      }}
      dragHandleClassName={styles.panelHeader}
    >
      <div className={styles.panelHeader}>{panel.title}</div>
      <div className={styles.panelContent}>
        {children}
      </div>
    </Rnd>
  );
};

// --- Define the default layouts FOR EACH TAB ---
const DEFAULT_LAYOUTS = {
  "Main Config": {
    'main': { id: 'main', title: 'Main Configuration', x: 50, y: 50, width: 800, height: 700, zIndex: 1 },
  },
  "Payload Workshop": {
    'payloads': { id: 'payloads', title: 'Payload Workshop', x: 350, y: 50, width: 700, height: 600, zIndex: 2 },
    'selection': { id: 'selection', title: 'Selected Payloads', x: 50, y: 50, width: 280, height: 450, zIndex: 3 }
  },
  "Optimization Guide": {
    'guide': { id: 'guide', title: 'Optimization Guide', x: 50, y: 50, width: 900, height: 700, zIndex: 1 },
  }
};

function Workbench() {
  const [activeTab, setActiveTab] = useState('Payload Workshop');
  const tabOptions = ['Main Config', 'Payload Workshop', 'Optimization Guide'];

  // State now holds layouts for ALL tabs
  const [layouts, setLayouts] = useState(() => {
    const saved = localStorage.getItem('workbench-layouts'); // plural!
    return saved ? JSON.parse(saved) : DEFAULT_LAYOUTS;
  });

  // Auto-save whenever any layout changes
  useEffect(() => {
    localStorage.setItem('workbench-layouts', JSON.stringify(layouts));
  }, [layouts]);
  
  const currentPanels = layouts[activeTab] || {};

  const handleLayoutChange = (panelId, position, size) => {
    setLayouts(prevLayouts => {
      const newLayouts = { ...prevLayouts };
      const currentTabLayout = { ...newLayouts[activeTab] };
      currentTabLayout[panelId] = { ...currentTabLayout[panelId], ...position, ...size };
      newLayouts[activeTab] = currentTabLayout;
      return newLayouts;
    });
  };

  const bringToFront = (panelId) => {
    const currentTabLayout = layouts[activeTab];
    const maxZ = Math.max(0, ...Object.values(currentTabLayout).map(p => p.zIndex));
    if (currentTabLayout[panelId].zIndex <= maxZ) {
        handleLayoutChange(panelId, { zIndex: maxZ + 1 });
    }
  };

  const handleResetLayout = () => {
      if (window.confirm(`Are you sure you want to reset the layout for the "${activeTab}" tab?`)) {
          setLayouts(prev => ({...prev, [activeTab]: DEFAULT_LAYOUTS[activeTab]}));
      }
  };

  const renderContentForPanel = (panelId) => {
    if (panelId === 'main') return <MainConfigTab />;
    if (panelId === 'payloads') return <PayloadsTab />;
    if (panelId === 'selection') return <p>Selection Box Content...</p>;
    if (panelId === 'guide') return <p>Optimization Guide Content...</p>;
    return null;
  }

  return (
    <div className={styles.pageContainer}>
      {/* The Permanent Sidebar */}
      <div className={styles.sidebar}>
          {/* We'll wrap the CustomSelect in a div that we can style */}
          <div className={styles.glassPanel}>
            <CustomSelect 
              options={tabOptions}
              value={activeTab}
              onChange={setActiveTab}
            />
          </div>
          <button onClick={handleResetLayout} className={styles.resetButton}>
            Reset View
          </button>
      </div>

      {/* The Workbench Canvas */}
      <div className={styles.workbenchCanvas}>
        {Object.values(currentPanels).map(panel => (
          <Window 
            key={panel.id}
            panel={panel} 
            onDragStop={(id, pos) => handleLayoutChange(id, pos)} 
            onResizeStop={(id, size, pos) => handleLayoutChange(id, {x: pos.x, y: pos.y}, {width: parseInt(size.width, 10), height: parseInt(size.height, 10)})}
            onBringToFront={bringToFront}
          >
            {renderContentForPanel(panel.id)}
          </Window>
        ))}
      </div>
    </div>
  );
}

export default Workbench;