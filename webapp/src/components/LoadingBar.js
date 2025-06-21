import React from 'react';

export default function LoadingBar({ progress }) {
  return (
    <div className="loading-container">
      <div className="loading-bar" style={{ width: `${progress}%` }} />
      <span>{progress}%</span>
    </div>
  );
}