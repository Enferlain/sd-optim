import React, { useState, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import styles from './Workbench.module.css';

import CustomSelect from './CustomSelect.js';
import MainConfigTab from './MainConfigTab.js';
import PayloadsTab from './PayloadsTab.js';

// --- A Reusable Window Component (with a new 'dragCancel' prop!) ---
const Window = ({ panel, onDragStop, onResizeStop, onBringToFront, children, dragCancelClassName }) => {
  return (
    <Rnd
      style={{ zIndex: panel.zIndex }}
      className={styles.glassPanel}
      size={{ width: panel.width, height: panel.height }}
      position={{ x: panel.x, y: panel.y }}
      minWidth={250}
      minHeight={60} // Allow smaller windows for things like the tab selector
      onDragStart={() => onBringToFront(panel.id)}
      onDragStop={(e, d) => onDragStop(panel.id, { x: d.x, y: d.y })}
      onResizeStart={() => onBringToFront(panel.id)}
      onResizeStop={(e, direction, ref, delta, position) => {
        onResizeStop(panel.id, { width: ref.style.width, height: ref.style.height }, position);
      }}
      // This is the magic! It makes the whole window draggable EXCEPT for elements with this class.
      cancel={`.${dragCancelClassName}`}
    >
      {/* We no longer need a separate header div! */}
      {children}
    </Rnd>
  );
};

// --- Define our default layout, now including the tab selector! ---
const DEFAULT_LAYOUTS = {
  // We'll have a base layout for shared components
  "base": {
    'selector': { id: 'selector', title: 'Navigation', x: 40, y: 40, width: 280, height: 70, zIndex: 99 },
  },
  "Main Config": {
    'main': { id: 'main', title: 'Main Configuration', x: 350, y: 40, width: 800, height: 700, zIndex: 1 },
  },
  "Payload Workshop": {
    'payloads': { id: 'payloads', title: 'Payload Workshop', x: 350, y: 40, width: 700, height: 600, zIndex: 2 },
    'selection': { id: 'selection', title: 'Selected Payloads', x: 40, y: 130, width: 280, height: 450, zIndex: 3 }
  },
  "Optimization Guide": {
    'guide': { id: 'guide', title: 'Optimization Guide', x: 350, y: 40, width: 900, height: 700, zIndex: 1 },
  }
};

function Workbench() {
  const [activeTab, setActiveTab] = useState('Payload Workshop');
  const tabOptions = ['Main Config', 'Payload Workshop', 'Optimization Guide'];

  const [layouts, setLayouts] = useState(() => {
    // ... (localStorage logic remains the same)
    const saved = localStorage.getItem('workbench-layouts-v2');
    return saved ? JSON.parse(saved) : DEFAULT_LAYOUTS;
  });

  useEffect(() => {
    localStorage.setItem('workbench-layouts-v2', JSON.stringify(layouts));
  }, [layouts]);
  
  // Combine the base layout with the active tab's layout
  const panelsToRender = { ...layouts.base, ...(layouts[activeTab] || {}) };

  const handleLayoutChange = (id, tab, position, size) => {
    setLayouts(prev => {
        const newLayouts = { ...prev };
        const targetTab = prev[tab];
        targetTab[id] = { ...targetTab[id], ...position, ...size };
        newLayouts[tab] = targetTab;
        return newLayouts;
    });
  };
  
  const createLayoutUpdater = (panelId, tabKey) => {
      const onDragStop = (id, pos) => handleLayoutChange(id, tabKey, pos);
      const onResizeStop = (id, size, pos) => handleLayoutChange(id, tabKey, { x: pos.x, y: pos.y }, { width: parseInt(size.width, 10), height: parseInt(size.height, 10) });
      const onBringToFront = (id) => { /* ... (z-index logic can be enhanced) ... */ };
      return { onDragStop, onResizeStop, onBringToFront };
  };

  const handleResetLayout = () => {
      if (window.confirm("Are you sure you want to reset ALL layouts to default?")) {
          setLayouts(DEFAULT_LAYOUTS);
          localStorage.removeItem('workbench-layouts-v2');
      }
  };

  const renderContentForPanel = (panelId) => {
      if (panelId === 'selector') return <CustomSelect options={tabOptions} value={activeTab} onChange={setActiveTab} />;
      if (panelId === 'main') return <MainConfigTab />;
      if (panelId === 'payloads') return <PayloadsTab />;
      if (panelId === 'selection') return <p>Selection Box Content...</p>;
      if (panelId === 'guide') return <p>Optimization Guide Content...</p>;
      return null;
  };

  return (
    <div className={styles.workbenchCanvas}>
      <button onClick={handleResetLayout} className={styles.resetButton} title="Reset Layout">
        {/* A nice SVG icon for the reset button! */}
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 0-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
      </button>

      {Object.values(panelsToRender).map(panel => {
        // Determine which layout group this panel belongs to
        const tabKey = Object.keys(layouts).find(key => key !== 'base' && panel.id in layouts[key]) || 'base';
        const updaters = createLayoutUpdater(panel.id, tabKey);
        
        return (
          <Window 
            key={panel.id}
            panel={panel} 
            {...updaters}
            // THIS IS THE KEY! Tell RND what class name to IGNORE for dragging.
            dragCancelClassName={styles.dragCancel} 
          >
            {/* The title is now part of the content, so we can style it and make it non-draggable */}
            <div className={`${styles.panelHeader} ${styles.dragCancel}`}>
              {panel.title}
            </div>
            <div className={`${styles.panelContent} ${styles.dragCancel}`}>
              {renderContentForPanel(panel.id)}
            </div>
          </Window>
        )
      })}
    </div>
  );
}

export default Workbench;