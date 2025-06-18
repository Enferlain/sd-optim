import React, { useState } from 'react';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css'; // Import the library's styles
import 'react-resizable/css/styles.css'; // And the resizable styles
import styles from './Workbench.module.css';

// Import our future panels
import MainConfigTab from './MainConfigTab.js';
import PayloadsTab from './PayloadsTab.js';

// A generic Panel wrapper
const Panel = ({ title, children }) => (
  <div className={styles.panel}>
    <div className={styles.panelHeader}>{title}</div>
    <div className={styles.panelContent}>{children}</div>
  </div>
);

function Workbench() {
  const [layout, setLayout] = useState([
    { i: 'a', x: 0, y: 0, w: 7, h: 12, minW: 4, minH: 6 }, // Main Config
    { i: 'b', x: 7, y: 0, w: 5, h: 8, minW: 3, minH: 4 }, // Payload Workshop
    { i: 'c', x: 7, y: 8, w: 5, h: 4, minW: 2, minH: 3 }  // Selection Box
  ]);

  return (
    <GridLayout
      className={styles.workbenchLayout}
      layout={layout}
      onLayoutChange={(newLayout) => setLayout(newLayout)}
      cols={12}
      rowHeight={30}
      width={1200}
      isDraggable={true}
      isResizable={true}
    >
      <div key="a">
        <Panel title="Main Configuration">
          <MainConfigTab />
        </Panel>
      </div>
      <div key="b">
        <Panel title="Payload Workshop">
          <PayloadsTab />
        </Panel>
      </div>
      <div key="c">
        <Panel title="Selected Payloads">
          {/* The selection box content goes here */}
          <p>Drag payloads here...</p>
        </Panel>
      </div>
    </GridLayout>
  );
}

export default Workbench;