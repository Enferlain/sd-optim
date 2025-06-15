import React, { useRef } from 'react';
import { useController } from 'react-hook-form'; // Make sure this import is here!
import styles from './MainConfigTab.module.css'; // We'll borrow styles

const FileInput = ({
    control,
    name,
    directory = false, // True for folder picker, false for file picker
    ...props // To pass down other props like 'placeholder'
}) => {
    const { field } = useController({ name, control });
    const fallbackInputRef = useRef(null); // Ref for the hidden input

    const handleButtonClick = async () => {
        // ---- The Modern API Path ----
        if ('showDirectoryPicker' in window) {
            try {
                if (directory) {
                    const directoryHandle = await window.showDirectoryPicker({
                        id: name, // Allows the browser to remember the last location for this specific input
                        mode: 'read',
                    });
                    // For now, we'll set the field value to the name of the chosen directory.
                    field.onChange(directoryHandle.name);
                } else {
                    const [fileHandle] = await window.showOpenFilePicker({ id: name });
                    field.onChange(fileHandle.name);
                }
            } catch (err) {
                // This error happens if the user clicks "Cancel", so we can safely ignore it.
                if (err.name !== 'AbortError') {
                    console.error('File System Access API error:', err);
                }
            }
        } else {
            // ---- Fallback for older browsers ----
            console.warn("File System Access API not supported. Using fallback input.");
            fallbackInputRef.current?.click();
        }
    };
    
    // This is for the fallback method only
    const handleFallbackChange = (event) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            field.onChange(files[0].name);
        }
    };

    return (
        <div style={{ display: 'flex', gap: 'var(--space-8)', width: '100%' }}>
            <input
                {...props} // Pass placeholder etc.
                value={field.value || ''}
                onChange={field.onChange}
                className={styles.input}
                style={{ flexGrow: 1 }}
            />
            <button
                type="button"
                className={styles.buttonSecondary}
                onClick={handleButtonClick}
            >
                {directory ? 'Browse...' : 'Select File...'}
            </button>

            {/* This input is hidden and only used as a fallback */}
            <input
                type="file"
                ref={fallbackInputRef}
                style={{ display: 'none' }}
                onChange={handleFallbackChange}
                {...(directory ? { webkitdirectory: "true" } : {})}
            />
        </div>
    );
};

export default FileInput;