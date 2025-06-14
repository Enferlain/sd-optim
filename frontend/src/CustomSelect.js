import React, { useState, useEffect, useRef } from 'react';
import styles from './CustomSelect.module.css';

const CustomSelect = ({ options, value, onChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const wrapperRef = useRef(null);

    // This is a clever trick to close the dropdown when you click anywhere else on the page!
    useEffect(() => {
        function handleClickOutside(event) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => { // Cleanup function to remove the listener when the component is gone
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [wrapperRef]);

    const handleOptionClick = (optionValue) => {
        onChange(optionValue);
        setIsOpen(false);
    };

    return (
        <div className={styles.selectWrapper} ref={wrapperRef}>
            <button type="button" className={styles.selectTrigger} onClick={() => setIsOpen(!isOpen)}>
                {value}
                <span className={`${styles.caret} ${isOpen ? styles.caretOpen : ''}`}>▼</span>
            </button>

            {isOpen && (
                <div className={styles.optionsMenu}>
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
            )}
        </div>
    );
};

export default CustomSelect;
