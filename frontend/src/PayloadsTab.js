import React, { useState, useEffect } from 'react';
import styles from './PayloadsTab.module.css';

// Placeholder for our future smart component
const GlyphCard = ({ name }) => (
  <div className={styles.glyphCard}>
    <div className={styles.glyphVisual}>◆</div>
    <p className={styles.glyphName}>{name}</p>
  </div>
);

function PayloadsTab() {
  const [payloads, setPayloads] = useState(['payload_one', 'payload_two', 'a_very_long_payload_name_that_tests_wrapping']);
  const [selectedPayloads, setSelectedPayloads] = useState(['payload_three']);

  return (
    <div className={styles.workshopContainer}>
      {/* Selection Box on the Left */}
      <div className={styles.selectionPanel}>
        <h3>Selected Payloads</h3>
        <div className={styles.selectionBox}>
            {selectedPayloads.map(name => <GlyphCard key={name} name={name} />)}
        </div>
      </div>

      {/* Main Workspace on the Right */}
      <div className={styles.workspacePanel}>
        <div className={styles.workspaceHeader}>
          <button className={styles.button}>+ Add New Payload</button>
          <input type="search" placeholder="Search payloads..." className={styles.searchInput} />
          {/* We can reuse the wiki button style from MainConfigTab if we make it global */}
          <button className={styles.button}>Wiki</button>
        </div>
        <div className={styles.workspaceCanvas}>
          {/* This is where the magic will happen! For now, a simple grid. */}
          {payloads.map(name => <GlyphCard key={name} name={name} />)}
        </div>
      </div>
    </div>
  );
}

export default PayloadsTab;