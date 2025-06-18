import React, { useState, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import styles from './Workbench.module.css';

import MainConfigTab from './MainConfigTab.js';
import PayloadsTab from './PayloadsTab.js';

// --- A Reusable Window Component (No changes here) ---
const Window = ({ panel, onDragStop, onResizeStop, onBringToFront, children }) => {
  return (
    <Rnd
      style={{ zIndex: panel.zIndex }} // Use zIndex from state
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
      dragHandleClassName={styles.panelHeader} // Make only the header draggable
    >
      <div className={styles.panelHeader}>{panel.title}</div>
      <div className={styles.panelContent}>
        {children}
      </div>
    </Rnd>
  );
};

// --- Define our default layout as a constant, so we can always refer back to it! ---
const DEFAULT_LAYOUT = {
  'main': { id: 'main', title: 'Main Configuration', x: 320, y: 40, width: 800, height: 700, zIndex: 1 },
  'payloads': { id: 'payloads', title: 'Payload Workshop', x: 350, y: 80, width: 700, height: 600, zIndex: 2 },
  'selection': { id: 'selection', title: 'Selected Payloads', x: 40, y: 40, width: 260, height: 400, zIndex: 3 }
};

// --- The Main Workbench Component ---
function Workbench() {
  // We now initialize our state by checking localStorage first!
  const [panels, setPanels] = useState(() => {
    try {
      const savedLayout = localStorage.getItem('workbench-layout');
      return savedLayout ? JSON.parse(savedLayout) : DEFAULT_LAYOUT;
    } catch (error) {
      console.error("Failed to parse saved layout, using default.", error);
      return DEFAULT_LAYOUT;
    }
  });

  // This `useEffect` hook is our auto-save feature! It runs whenever `panels` changes.
  useEffect(() => {
    try {
      localStorage.setItem('workbench-layout', JSON.stringify(panels));
    } catch (error) {
      console.error("Failed to save layout.", error);
    }
  }, [panels]);

  const handleDragStop = (id, position) => {
    setPanels(prev => ({ ...prev, [id]: { ...prev[id], ...position } }));
  };

  const handleResizeStop = (id, size, position) => {
    const newWidth = parseInt(size.width, 10);
    const newHeight = parseInt(size.height, 10);
    setPanels(prev => ({ ...prev, [id]: { ...prev[id], width: newWidth, height: newHeight, ...position } }));
  };
  
  // This function brings a clicked/dragged window to the front
  const bringToFront = (id) => {
      setPanels(prev => {
          const maxZ = Math.max(...Object.values(prev).map(p => p.zIndex));
          if (prev[id].zIndex <= maxZ) {
              return {...prev, [id]: { ...prev[id], zIndex: maxZ + 1 }};
          }
          return prev;
      });
  };

  // --- Our new "Emergency Reset" function! ---
  const handleResetLayout = () => {
    if (window.confirm("Are you sure you want to reset the layout to default?")) {
      setPanels(DEFAULT_LAYOUT);
      // We remove the saved layout so the default loads on next refresh too
      localStorage.removeItem('workbench-layout');
    }
  };

  return (
    <div className={styles.workbenchCanvas}>
      {/* Our new Reset Button! */}
      <button onClick={handleResetLayout} className={styles.resetButton}>
        Reset Layout
      </button>

      {Object.values(panels).map(panel => {
        let content;
        // A simple way to map content based on panel ID
        if (panel.id === 'main') content = <MainConfigTab />;
        else if (panel.id === 'payloads') content = <PayloadsTab />;
        else if (panel.id === 'selection') content = <p>Selection Box Content</p>;
        else content = <p>Empty Panel</p>;

        return (
          <Window 
            key={panel.id}
            panel={panel} 
            onDragStop={handleDragStop} 
            onResizeStop={handleResizeStop}
            onBringToFront={bringToFront}
          >
            {content}
          </Window>
        )
      })}
    </div>
  );
}

export default Workbench;