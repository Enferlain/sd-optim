import React, { useState } from 'react';
import styles from './PayloadsTab.module.css';

// We can define the placeholder here for now
const GlyphCard = ({ name }) => (
    <div className={styles.glyphCard}>
      <div className={styles.glyphVisual}>◆</div>
      <p className={styles.glyphName}>{name}</p>
    </div>
);

function PayloadsTab() {
  // This component will eventually get the list of payloads from its parent
  const [unselectedPayloads, setUnselectedPayloads] = useState(['payload_one', 'payload_two', 'a_very_long_payload_name_that_tests_wrapping']);

  return (
    <div className={styles.workspaceContainer}>
      <div className={styles.workspaceHeader}>
        <button className={styles.button}>+ Add New Payload</button>
        <input type="search" placeholder="Search payloads..." className={styles.searchInput} />
        <button className={styles.button}>Wiki</button>
      </div>
      <div className={styles.workspaceCanvas}>
        {unselectedPayloads.map(name => <GlyphCard key={name} name={name} />)}
      </div>
    </div>
  );
}

export default PayloadsTab;