import React, { useState, useRef } from 'react';
import { useLocation } from 'react-router-dom';

export default function AudioReportPage() {
  const location = useLocation();
  const { reportData } = location.state || {};
  
  // Audio playback states
  const [isPlayingEnhanced, setIsPlayingEnhanced] = useState(false);
  const [currentView, setCurrentView] = useState('both');
  
  const enhancedAudioRef = useRef(null);
  const [audioError, setAudioError] = useState('');

  // Safe helper functions
  const safeString = (value, defaultValue = '') => {
    if (value === null || value === undefined) return defaultValue;
    return String(value);
  };

  const safeNumber = (value, defaultValue = 0) => {
    if (value === null || value === undefined || isNaN(value)) return defaultValue;
    return Number(value);
  };

  const safeArray = (value, defaultValue = []) => {
    if (!Array.isArray(value)) return defaultValue;
    return value;
  };

  const safeObject = (value, defaultValue = {}) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return defaultValue;
    return value;
  };

  const styles = {
    container: {
      maxWidth: '1400px',
      margin: '0 auto',
      padding: '2rem',
      backgroundColor: '#f9f9ff',
      minHeight: '100vh',
      fontFamily: 'system-ui, sans-serif'
    },
    header: {
      textAlign: 'center',
      marginBottom: '3rem'
    },
    title: {
      fontSize: '2.8rem',
      color: '#5D2E8C',
      marginBottom: '0.5rem',
      fontWeight: '700'
    },
    subtitle: {
      fontSize: '1.3rem',
      color: '#666',
      marginBottom: '2rem'
    },
    metaInfo: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1.5rem',
      marginBottom: '3rem',
      maxWidth: '1000px',
      margin: '0 auto 3rem auto'
    },
    metaItem: {
      backgroundColor: 'white',
      padding: '1.5rem',
      borderRadius: '1.2rem',
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
      textAlign: 'center',
      border: '1px solid #e0e0e0'
    },
    metaLabel: {
      fontSize: '0.9rem',
      color: '#666',
      marginBottom: '0.5rem',
      fontWeight: '500'
    },
    metaValue: {
      fontSize: '1.4rem',
      fontWeight: 'bold',
      color: '#5D2E8C'
    },
    scoreSection: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '2rem',
      marginBottom: '3rem',
      maxWidth: '1200px',
      margin: '0 auto 3rem auto'
    },
    scoreCard: {
      backgroundColor: 'white',
      padding: '2rem',
      borderRadius: '1.5rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      textAlign: 'center',
      border: '2px solid #e0e0e0'
    },
    scoreNumber: {
      fontSize: '3rem',
      fontWeight: 'bold',
      marginBottom: '0.5rem'
    },
    scoreLabel: {
      fontSize: '1.1rem',
      color: '#666',
      fontWeight: '600'
    },
    viewToggle: {
      display: 'flex',
      justifyContent: 'center',
      gap: '1rem',
      marginBottom: '2rem'
    },
    toggleButton: {
      padding: '0.8rem 1.5rem',
      borderRadius: '1rem',
      border: 'none',
      cursor: 'pointer',
      fontSize: '1rem',
      fontWeight: '600',
      transition: 'all 0.3s ease'
    },
    toggleButtonActive: {
      backgroundColor: '#5D2E8C',
      color: 'white',
      transform: 'translateY(-2px)',
      boxShadow: '0 4px 12px rgba(93, 46, 140, 0.3)'
    },
    toggleButtonInactive: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      border: '2px solid #5D2E8C'
    },
    transcriptGrid: {
      display: 'grid',
      gridTemplateColumns: currentView === 'both' ? '1fr 1fr' : '1fr',
      gap: '2rem',
      marginBottom: '3rem'
    },
    transcriptCard: {
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0'
    },
    cardTitle: {
      fontSize: '1.5rem',
      fontWeight: 'bold',
      color: '#5D2E8C',
      marginBottom: '1.5rem'
    },
    transcript: {
      backgroundColor: '#f8f9fa',
      padding: '1.5rem',
      borderRadius: '1rem',
      border: '1px solid #e9ecef',
      maxHeight: '350px',
      overflowY: 'auto',
      fontSize: '1rem',
      lineHeight: '1.7',
      color: '#333',
      whiteSpace: 'pre-wrap',
      marginBottom: '1rem'
    },
    audioControls: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      marginTop: '1rem',
      flexWrap: 'wrap'
    },
    playButton: {
      padding: '0.8rem 1.5rem',
      borderRadius: '2rem',
      border: 'none',
      cursor: 'pointer',
      fontSize: '1rem',
      fontWeight: '600',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      transition: 'all 0.3s ease',
      textDecoration: 'none'
    },
    playButtonEnhanced: {
      backgroundColor: '#5D2E8C',
      color: 'white'
    },
    downloadButton: {
      backgroundColor: '#17a2b8',
      color: 'white'
    },
    enhancementCard: {
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0',
      marginBottom: '2rem'
    },
    enhancementHighlight: {
      backgroundColor: '#e8f5e8',
      padding: '1.5rem',
      borderRadius: '1rem',
      border: '1px solid #c3e6c3',
      marginTop: '1rem'
    },
    improvementsList: {
      backgroundColor: '#f8f9fa',
      padding: '1.5rem',
      borderRadius: '1rem',
      border: '1px solid #e9ecef',
      marginTop: '1rem'
    },
    improvementItem: {
      padding: '0.8rem 0',
      borderBottom: '1px solid #dee2e6',
      fontSize: '1rem',
      display: 'flex',
      alignItems: 'flex-start',
      gap: '0.8rem'
    },
    speakingTipsCard: {
      backgroundColor: '#fff3cd',
      border: '1px solid #ffeaa7',
      borderRadius: '1.5rem',
      padding: '2rem',
      marginBottom: '2rem'
    },
    tipItem: {
      padding: '0.8rem 0',
      fontSize: '1rem',
      display: 'flex',
      alignItems: 'flex-start',
      gap: '0.8rem',
      color: '#856404'
    },
    presentationMetricsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '2rem',
      marginBottom: '3rem'
    },
    metricCard: {
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0'
    },
    metricHeader: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      marginBottom: '1.5rem'
    },
    metricIcon: {
      fontSize: '2rem'
    },
    metricTitle: {
      fontSize: '1.3rem',
      fontWeight: 'bold',
      color: '#5D2E8C'
    },
    feedback: {
      backgroundColor: '#f8f9fa',
      padding: '1rem',
      borderRadius: '0.75rem',
      fontSize: '0.95rem',
      lineHeight: '1.6',
      color: '#333',
      border: '1px solid #e9ecef'
    },
    errorMessage: {
      backgroundColor: '#f8d7da',
      color: '#721c24',
      padding: '1rem',
      borderRadius: '0.5rem',
      border: '1px solid #f5c6cb',
      marginTop: '1rem'
    },
    noDataMessage: {
      textAlign: 'center',
      padding: '3rem',
      fontSize: '1.2rem',
      color: '#666'
    }
  };

  if (!reportData) {
    return (
      <div style={styles.container}>
        <div style={styles.noDataMessage}>
          <h1>No audio analysis data found</h1>
          <p>Please upload an audio file for analysis.</p>
        </div>
      </div>
    );
  }

  // COMPLETELY SAFE data extraction with bulletproof fallbacks
  const transcriptSegments = safeString(reportData.transcriptSegments, "No transcript available");
  const enhancedTranscript = safeString(reportData.enhancedTranscript, null);
  const enhancedAudioUrl = safeString(reportData.enhancedAudioUrl, null);
  const duration = safeNumber(reportData.duration, 0);
  const language = safeString(reportData.language, "en");
  const overallScore = safeNumber(reportData.overallScore, 0);

  // Safe nested object access
  const presentationMetrics = safeObject(reportData.presentationMetrics);
  const enhancement = safeObject(reportData.enhancement);
  const speechAnalysis = safeObject(reportData.speechAnalysis);

  // Individual metric scores with safe access
  const clarityScore = safeNumber(presentationMetrics.clarity_score, 0);
  const paceScore = safeNumber(presentationMetrics.pace_score, 0);
  const confidenceScore = safeNumber(presentationMetrics.confidence_score, 0);
  const engagementScore = safeNumber(presentationMetrics.engagement_score, 0);

  // Safe feedback strings
  const clarityFeedback = safeString(presentationMetrics.clarity_feedback, 'Focus on making your message clear and well-structured.');
  const paceFeedback = safeString(presentationMetrics.pace_feedback || speechAnalysis.pace_feedback, 'Work on maintaining an appropriate speaking pace.');
  const confidenceFeedback = safeString(presentationMetrics.confidence_feedback || speechAnalysis.filler_feedback, 'Focus on reducing filler words and speaking with conviction.');
  const engagementFeedback = safeString(presentationMetrics.engagement_feedback, 'Add more energy and enthusiasm to capture audience attention.');

  // Safe arrays
  const keyChanges = safeArray(enhancement.key_changes);
  const speakingTips = safeArray(enhancement.speaking_tips);

  // Safe speech analysis values
  const wordCount = safeNumber(speechAnalysis.word_count, 0);
  const speakingRate = safeNumber(speechAnalysis.speaking_rate, 0);

  // Safe helper functions for display
  const getScoreColor = (score) => {
    const safeScore = safeNumber(score, 0);
    if (safeScore >= 80) return '#28a745';
    if (safeScore >= 60) return '#ffc107'; 
    return '#dc3545';
  };

  const formatDuration = (seconds) => {
    const safeSeconds = safeNumber(seconds, 0);
    const mins = Math.floor(safeSeconds / 60);
    const secs = Math.floor(safeSeconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Completely safe language display
  const safeLanguage = safeString(language, 'en').toUpperCase();

  // Audio playback handlers with safe checks
  const handlePlayEnhanced = () => {
    if (enhancedAudioRef.current && enhancedAudioUrl) {
      if (isPlayingEnhanced) {
        enhancedAudioRef.current.pause();
      } else {
        enhancedAudioRef.current.play().catch(error => {
          console.error('Audio playback failed:', error);
          setAudioError('Failed to play audio. The file might not be available yet.');
        });
      }
    }
  };

  const handleDownload = () => {
    if (enhancedAudioUrl) {
      const link = document.createElement('a');
      link.href = `http://localhost:4000${enhancedAudioUrl}`;
      link.download = `enhanced_speech_${Date.now()}.mp3`;
      link.click();
    }
  };

  const renderViewToggle = () => (
    <div style={styles.viewToggle}>
      <button
        style={{
          ...styles.toggleButton,
          ...(currentView === 'original' ? styles.toggleButtonActive : styles.toggleButtonInactive)
        }}
        onClick={() => setCurrentView('original')}
      >
        Original Only
      </button>
      <button
        style={{
          ...styles.toggleButton,
          ...(currentView === 'both' ? styles.toggleButtonActive : styles.toggleButtonInactive)
        }}
        onClick={() => setCurrentView('both')}
      >
        Side by Side
      </button>
      <button
        style={{
          ...styles.toggleButton,
          ...(currentView === 'enhanced' ? styles.toggleButtonActive : styles.toggleButtonInactive)
        }}
        onClick={() => setCurrentView('enhanced')}
      >
        Enhanced Only
      </button>
    </div>
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Presentation Speech Analysis</h1>
        <p style={styles.subtitle}>AI-Powered Speech Coaching & Enhancement</p>
        
        {/* Meta Information */}
        <div style={styles.metaInfo}>
          <div style={styles.metaItem}>
            <div style={styles.metaLabel}>Duration</div>
            <div style={styles.metaValue}>{formatDuration(duration)}</div>
          </div>
          <div style={styles.metaItem}>
            <div style={styles.metaLabel}>Language</div>
            <div style={styles.metaValue}>{safeLanguage}</div>
          </div>
          <div style={styles.metaItem}>
            <div style={styles.metaLabel}>Word Count</div>
            <div style={styles.metaValue}>{wordCount}</div>
          </div>
          <div style={styles.metaItem}>
            <div style={styles.metaLabel}>Speaking Rate</div>
            <div style={styles.metaValue}>{speakingRate} WPM</div>
          </div>
        </div>

        {/* Presentation Scores */}
        <div style={styles.scoreSection}>
          <div style={styles.scoreCard}>
            <div style={{...styles.scoreNumber, color: getScoreColor(clarityScore)}}>
              {clarityScore}
            </div>
            <div style={styles.scoreLabel}>Clarity Score</div>
          </div>
          <div style={styles.scoreCard}>
            <div style={{...styles.scoreNumber, color: getScoreColor(paceScore)}}>
              {paceScore}
            </div>
            <div style={styles.scoreLabel}>Pace Score</div>
          </div>
          <div style={styles.scoreCard}>
            <div style={{...styles.scoreNumber, color: getScoreColor(confidenceScore)}}>
              {confidenceScore}
            </div>
            <div style={styles.scoreLabel}>Confidence Score</div>
          </div>
          <div style={styles.scoreCard}>
            <div style={{...styles.scoreNumber, color: getScoreColor(engagementScore)}}>
              {engagementScore}
            </div>
            <div style={styles.scoreLabel}>Engagement Score</div>
          </div>
          <div style={styles.scoreCard}>
            <div style={{...styles.scoreNumber, color: getScoreColor(overallScore)}}>
              {overallScore}
            </div>
            <div style={styles.scoreLabel}>Overall Score</div>
          </div>
        </div>
      </div>

      {/* Enhanced Speech Section */}
      {enhancedTranscript && (
        <div style={styles.enhancementCard}>
          <h3 style={styles.cardTitle}>AI-Enhanced Speech</h3>
          <div style={styles.feedback}>
            {safeString(enhancement.summary, 'Your speech has been enhanced for better presentation delivery.')}
          </div>
          
          {/* Enhanced Audio Player */}
          {enhancedAudioUrl && (
            <div style={styles.enhancementHighlight}>
              <h4 style={{color: '#2d5a2d', marginBottom: '1rem', fontSize: '1.1rem'}}>
                Listen to Your Enhanced Speech
              </h4>
              <div style={styles.audioControls}>
                <button
                  style={{...styles.playButton, ...styles.playButtonEnhanced}}
                  onClick={handlePlayEnhanced}
                >
                  {isPlayingEnhanced ? '⏸️ Pause Enhanced' : 'Play Enhanced Speech'}
                </button>
                
                <button
                  style={{...styles.playButton, ...styles.downloadButton}}
                  onClick={handleDownload}
                >
                  Download Enhanced Audio
                </button>
              </div>
              
              {audioError && (
                <div style={styles.errorMessage}>{audioError}</div>
              )}
              
              {/* Enhanced Audio Element */}
              <audio
                ref={enhancedAudioRef}
                controls
                style={{width: '100%', marginTop: '1rem'}}
                onPlay={() => setIsPlayingEnhanced(true)}
                onPause={() => setIsPlayingEnhanced(false)}
                onEnded={() => setIsPlayingEnhanced(false)}
                onError={() => setAudioError('Audio file could not be loaded. Please try again later.')}
              >
                <source src={`http://localhost:4000${enhancedAudioUrl}`} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
          
          {/* Key Changes Made */}
          {keyChanges.length > 0 && (
            <div style={styles.improvementsList}>
              <h4 style={{margin: '0 0 1rem 0', color: '#5D2E8C'}}>Key Improvements Made:</h4>
              {keyChanges.map((change, index) => (
                <div key={index} style={styles.improvementItem}>
                  <span style={{color: '#28a745', fontSize: '1.2rem', fontWeight: 'bold'}}>✓</span>
                  <span>{safeString(change, 'Improvement made')}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Speaking Tips */}
      {speakingTips.length > 0 && (
        <div style={styles.speakingTipsCard}>
          <h3 style={{color: '#856404', marginBottom: '1rem', fontSize: '1.4rem'}}>
            Personalized Speaking Tips
          </h3>
          {speakingTips.map((tip, index) => (
            <div key={index} style={styles.tipItem}>
              <span style={{fontSize: '1.2rem'}}>- </span>
              <span>{safeString(tip, 'Speaking tip')}</span>
            </div>
          ))}
        </div>
      )}

      {/* View Toggle */}
      {enhancedTranscript && renderViewToggle()}

      {/* Transcript Comparison */}
      <div style={styles.transcriptGrid}>
        {(currentView === 'original' || currentView === 'both') && (
          <div style={styles.transcriptCard}>
            <div style={styles.cardTitle}>Original Transcript</div>
            <div style={styles.transcript}>{transcriptSegments}</div>
          </div>
        )}

        {enhancedTranscript && (currentView === 'enhanced' || currentView === 'both') && (
          <div style={styles.transcriptCard}>
            <div style={styles.cardTitle}>Enhanced Transcript</div>
            <div style={styles.transcript}>{enhancedTranscript}</div>
          </div>
        )}
      </div>

      {/* Detailed Presentation Metrics */}
      <div style={styles.presentationMetricsGrid}>
        <div style={styles.metricCard}>
          <div style={styles.metricHeader}>
            <span style={styles.metricIcon}></span>
            <span style={styles.metricTitle}>Clarity Analysis</span>
          </div>
          <div style={styles.feedback}>{clarityFeedback}</div>
        </div>

        <div style={styles.metricCard}>
          <div style={styles.metricHeader}>
            <span style={styles.metricIcon}></span>
            <span style={styles.metricTitle}>Pace Analysis</span>
          </div>
          <div style={styles.feedback}>{paceFeedback}</div>
        </div>

        <div style={styles.metricCard}>
          <div style={styles.metricHeader}>
            <span style={styles.metricIcon}></span>
            <span style={styles.metricTitle}>Confidence Analysis</span>
          </div>
          <div style={styles.feedback}>{confidenceFeedback}</div>
        </div>

        <div style={styles.metricCard}>
          <div style={styles.metricHeader}>
            <span style={styles.metricIcon}></span>
            <span style={styles.metricTitle}>Engagement Analysis</span>
          </div>
          <div style={styles.feedback}>{engagementFeedback}</div>
        </div>
      </div>
    </div>
  );
}
