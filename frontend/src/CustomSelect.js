import React, { useState, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom'; // <-- ADD THIS IMPORT!
import styles from './CustomSelect.module.css';

const CustomSelect = ({ options, value, onChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const wrapperRef = useRef(null);

    // This effect now needs to calculate the dropdown's position
    const [menuPosition, setMenuPosition] = useState({ top: 0, left: 0, width: 0 });

    useEffect(() => {
        if (isOpen && wrapperRef.current) {
            const rect = wrapperRef.current.getBoundingClientRect();
            setMenuPosition({
                top: rect.bottom + 4, // Position it right below the select box
                left: rect.left,
                width: rect.width
            });
        }
    }, [isOpen]);


    useEffect(() => {
        function handleClickOutside(event) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [wrapperRef]);

    const handleOptionClick = (optionValue) => {
        onChange(optionValue);
        setIsOpen(false);
    };

    // The new dropdown menu component that will be portaled
    const DropdownMenu = () => (
        <div 
          className={styles.optionsMenu} 
          style={{ 
            top: `${menuPosition.top}px`, 
            left: `${menuPosition.left}px`, 
            width: `${menuPosition.width}px` 
          }}
        >
            {options.map((option) => (
                <div
                    key={option}
                    className={`${styles.optionItem} ${option === value ? styles.optionSelected : ''}`}
                    onClick={() => handleOptionClick(option)}
                >
                    {option}
                </div>
            ))}
        </div>
    );

    return (
        <div className={styles.selectWrapper} ref={wrapperRef}>
            <button type="button" className={styles.selectTrigger} onClick={() => setIsOpen(!isOpen)}>
                {value}
                <span className={`${styles.caret} ${isOpen ? styles.caretOpen : ''}`}>▼</span>
            </button>

            {/* Use the portal to render the dropdown menu at the root of the document */}
            {isOpen && ReactDOM.createPortal(<DropdownMenu />, document.body)}
        </div>
    );
};

export default CustomSelect;