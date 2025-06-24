import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip
} from 'recharts';
import html2canvas from 'html2canvas';    
import { useAuth } from '../context/AuthContext';
import { FaDownload } from 'react-icons/fa';

export default function AudioReportPage() {
  const location = useLocation();
  const [reportData, setReportData] = useState(location.state?.reportData || null);
  const [dataSource, setDataSource] = useState('direct'); // 'direct' or 'video'
  const [isLoading, setIsLoading] = useState(!location.state?.reportData);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingMessage, setLoadingMessage] = useState('Initializing analysis...');
  const [analysisError, setAnalysisError] = useState(null);

  const { user } = useAuth();
  const username = user?.email?.split('@')[0] || 'Guest';

  const date = new Date().toLocaleDateString('en-AU', {
    weekday: 'short',
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });

  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const chartWidth = isMobile ? 300 : 400;
  const chartHeight = isMobile ? 240 : 400;
  const outerRadius = isMobile ? 80 : 120;

  // Audio playback states
  const [isPlayingEnhanced, setIsPlayingEnhanced] = useState(false);
  const [currentView, setCurrentView] = useState('both');

  const enhancedAudioRef = useRef(null);
  const [audioError, setAudioError] = useState('');
  const BASE_URL = process.env.REACT_APP_API_HOST;

  // Define styles at the top to avoid "used before defined" warnings
  const styles = {
    container: {
      backgroundColor: '#f9f9f9',
      minHeight: '100vh',
      padding: '20px',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    },
    reportBoxOuter: {
      backgroundColor: '#5D2E8C',
      borderRadius: '20px',
      padding: '10px',
      maxWidth: '1200px',
      margin: '0 auto'
    },
    reportBoxInner: {
      border: '2px dotted white',
      borderRadius: '15px',
      padding: '20px'
    },
    header: {
      textAlign: 'center',
      marginBottom: '3rem'
    },
    title: {
      textAlign: 'center',
      fontSize: '36px',
      marginBottom: '20px',
      color: 'white',
      fontWeight: '600',
      letterSpacing: '1px'
    },
    divider: {
      borderTop: '2px solid white',
      margin: '20px 0'
    },
    subtitle: {
      fontSize: '1.3rem',
      color: '#666',
      marginBottom: '2rem'
    },
    // NEW: Source indicator styles
    sourceIndicator: {
      display: 'inline-block',
      backgroundColor: dataSource === 'video' ? '#e8f5e8' : '#f0f8ff',
      border: `2px solid ${dataSource === 'video' ? '#28a745' : '#007bff'}`,
      borderRadius: '1rem',
      padding: '1rem 2rem',
      marginBottom: '2rem',
      textAlign: 'center',
      maxWidth: '600px',
      margin: '0 auto 2rem auto',
      color: dataSource === 'video' ? '#155724' : '#004085'
    },
    sourceText: {
      fontSize: '1.1rem',
      fontWeight: '600',
      margin: 0
    },
    sourceSubtext: {
      fontSize: '0.95rem',
      marginTop: '0.5rem',
      margin: '0.5rem 0 0 0'
    },

    reportContainer: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '20px',
      justifyContent: 'center'
    },
    leftPanel: {
      backgroundColor: '#fff',
      borderRadius: '15px',
      minWidth: '50px',
      justifyContent: 'center',
      justifyItems: 'center',
      flex: '1 1 400px'
    },
    rightPanel: {
      flexDirection: 'column',
      gap: '20px',
      flex: '1 1 300px',
      display: 'grid',
    },
    metaInfo: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, max(1fr))',
      gap: '1.5rem',
      marginBottom: '3rem',
      maxWidth: '1000px',
      margin: '0 auto 3rem auto'
    },
    metaItem: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      borderRadius: '15px',
      padding: '20px',
      textAlign: 'center'
    },
    metaLabel: {
      fontSize: '12px',
      textTransform: 'uppercase',
      fontWeight: '500'
    },
    metaValue: {
      fontSize: '50px',
      fontWeight: '1000',
      fontStyle: 'italic'
    },
    userInfoCard: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '20px'
    },
    userInfoValue: {
      fontSize: '18px',
      fontWeight: '700',
      marginBottom: '10px'
    },
    scoreRadarContainer: {
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'center',
      gap: '2rem',
      maxWidth: '1200px',
      margin: '0 auto 3rem auto'
    },
    radarCard: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0',
      textAlign: 'center',
      flex: '1 1 350px'
    },
    radarTitle: {
      fontSize: '1.3rem',
      color: '#5D2E8C',
      fontWeight: '600',
      marginBottom: '1rem'
    },
    scoreSection: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, max(1fr))', // Increased from 200px to ensure better fit
      gap: '1.5rem', // Reduced gap slightly
      marginBottom: '3rem',
      maxWidth: '1200px',
      margin: '0 auto 3rem auto'
    },
    scoreCard: {
      backgroundColor: 'white',
      padding: '1.5rem',
      borderRadius: '1.5rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      textAlign: 'center',
      border: '2px solid #e0e0e0',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center'
    },
    scoreNumber: {
      fontSize: '3rem',
      fontWeight: 'bold',
      marginBottom: '0.5rem'
    },
    scoreLabel: {
      fontSize: '1.1rem',
      color: '#666',
      fontWeight: '600',
      wordWrap: 'break-word',
      textAlign: 'center',
      lineHeight: '1.2'
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
      fontSize: '.7rem',
      fontWeight: '600',
      transition: 'all 0.3s ease'
    },
    toggleButtonActive: {
      backgroundColor: '#5D2E8C',
      color: 'white',
      border: '2px solid white',
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
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
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
      textAlign: 'center',
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
      gridTemplateColumns: isMobile
      ? '1fr'
      : 'repeat(2, 1fr)',
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
    },
    vocalDynamicsSection: {
      marginBottom: '3rem'
    },

    vocalDynamicsHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '2rem',
      padding: '2rem',
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0'
    },

    sectionTitle: {
      fontSize: '2rem',
      color: '#5D2E8C',
      fontWeight: 'bold',
      margin: 0
    },

    overallDynamicsScore: {
      textAlign: 'center'
    },

    readinessLevel: {
      fontSize: '1rem',
      color: '#666',
      marginTop: '0.5rem',
      fontWeight: '500'
    },

    // Score range legend styles (keeping the fixed ones from before)
    scoreRangeLegend: {
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0',
      marginBottom: '2rem'
    },

    legendTitle: {
      fontSize: '1.4rem',
      color: '#5D2E8C',
      fontWeight: 'bold',
      marginBottom: '1.5rem',
      textAlign: 'center'
    },

    legendGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem'
    },

    legendItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
      padding: '0.5rem'
    },

    legendColor: {
      width: '16px',
      height: '16px',
      borderRadius: '3px',
      flexShrink: 0
    },

    legendText: {
      fontSize: '0.95rem',
      color: '#333',
      fontWeight: '500'
    },

    vocalDynamicsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '2rem',
      marginBottom: '2rem'
    },

    vocalCard: {
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0',
      transition: 'transform 0.2s ease, box-shadow 0.2s ease'
    },

    vocalCardHeader: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      marginBottom: '1.5rem'
    },

    iconContainer: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: '3rem',
      height: '3rem',
      borderRadius: '0.75rem',
      backgroundColor: 'rgba(93, 46, 140, 0.1)',
      flexShrink: 0
    },

    cardIcon: {
      width: '1.5rem',
      height: '1.5rem',
      color: '#5D2E8C'
    },

    vocalIcon: {
      fontSize: '2.5rem',
      minWidth: '3rem'
    },

    vocalCardTitle: {
      fontSize: '1.3rem',
      color: '#5D2E8C',
      fontWeight: 'bold',
      margin: '0 0 0.5rem 0'
    },

    vocalScore: {
      fontSize: '1.5rem',
      fontWeight: 'bold'
    },

    scoreRangeIndicator: {
      fontSize: '0.9rem',
      color: '#666',
      fontWeight: '500',
      marginTop: '0.25rem'
    },

    vocalCardContent: {
      display: 'flex',
      flexDirection: 'column',
      gap: '0.8rem'
    },

    vocalMetric: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '0.5rem 0',
      borderBottom: '1px solid #f0f0f0'
    },

    metricLabel: {
      fontSize: '0.95rem',
      color: '#666',
      fontWeight: '500'
    },

    metricValue: {
      fontSize: '1rem',
      color: '#333',
      fontWeight: 'bold'
    },

    improvementTip: {
      backgroundColor: '#f0f8ff',
      padding: '1rem',
      borderRadius: '0.75rem',
      border: '1px solid #d1ecf1',
      marginTop: '1rem',
      fontSize: '0.9rem',
      lineHeight: '1.5',
      color: '#0c5460'
    },

    vocalSummaryCard: {
      backgroundColor: '#f8f9fa',
      borderRadius: '1.5rem',
      padding: '2rem',
      border: '1px solid #e9ecef'
    },

    vocalSummaryContent: {
      fontSize: '1.1rem',
      lineHeight: '1.6',
      color: '#333',
      fontStyle: 'italic'
    },

    benchmarkCard: {
      backgroundColor: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 6px 20px rgba(0,0,0,0.1)',
      border: '1px solid #e0e0e0',
      marginBottom: '2rem'
    },

    benchmarkContent: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '1rem'
    },

    benchmarkItem: {
      display: 'flex',
      width: '75%',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '1rem',
      backgroundColor: '#f8f9fa',
      borderRadius: '0.75rem',
      border: '1px solid #e9ecef'
    },

    benchmarkLabel: {
      fontSize: '.8rem',
      color: '#666',
      fontWeight: '500'
    },

    benchmarkValue: {
      fontSize: '1rem',
      color: '#5D2E8C',
      fontWeight: 'bold'
    },
      progressBar: {
    height: '100%',
    backgroundSize: '40px 40px',
    backgroundPosition: '0 0',
    animation: 'candy 1s linear infinite',
    transition: 'width 1s ease-out',
  },
  };

  const [isExporting, setIsExporting] = useState(false);       
  const reportRef   = useRef(null);
const handlePrint = async () => {
    if (!reportRef.current) return;

    setIsExporting(true);                          // 1️⃣ flag on
    // give React a tick to paint the “export mode” CSS
    await new Promise(r => setTimeout(r, 50));

    reportRef.current.classList.add('screenshot-mode');

    try {
      const canvas = await html2canvas(reportRef.current, {
        scale: 1,
        useCORS: true
      });
      const img  = canvas.toDataURL('image/png');
      const a    = document.createElement('a');
      a.href      = img;
      a.download  = 'Performance_Report.png';
      a.click();
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      reportRef.current.classList.remove('screenshot-mode');
      setIsExporting(false);                       // 2️⃣ flag off
    }
  };

  // Helper functions for vocal improvement tips
  const getPitchImprovementTip = (score) => {
    if (score >= 90) return 'You\'re at professional level! Maintain consistency across longer presentations.';
    if (score >= 80) return 'Add more dramatic pitch changes for emphasis. Target 90+ for outstanding level.';
    if (score >= 60) return 'Practice reading with exaggerated expression. Aim for 80+ excellent level.';
    if (score >= 40) return 'Focus on varying your pitch more. Good speakers use 20-30% pitch variation.';
    return 'Start with vocal warm-ups and practice speaking with more emotion.';
  };

  const getVolumeImprovementTip = (score) => {
    if (score >= 90) return 'Perfect energy control! You\'re ready for any presentation venue.';
    if (score >= 80) return 'Work on maintaining energy throughout longer speeches. Target 90+ for outstanding.';
    if (score >= 60) return 'Practice breath support exercises to boost energy sustainability.';
    if (score >= 40) return 'Focus on projecting your voice and using more dynamic volume changes.';
    return 'Start with diaphragmatic breathing exercises and practice speaking louder.';
  };

  const getRhythmImprovementTip = (score) => {
    if (score >= 90) return 'Excellent rhythm control! You speak like a seasoned professional.';
    if (score >= 80) return 'Fine-tune your pacing for different content types. Target 90+ for outstanding.';
    if (score >= 60) return 'Practice with a metronome to develop more consistent rhythm.';
    if (score >= 40) return 'Work on speaking at a steady pace - not too fast, not too slow.';
    return 'Focus on basic pacing: 140-160 words per minute is optimal.';
  };

  const getPauseImprovementTip = (score) => {
    if (score >= 90) return 'Masterful pause placement! You use silence as effectively as words.';
    if (score >= 80) return 'Experiment with longer pauses for dramatic effect. Target 90+ for outstanding.';
    if (score >= 60) return 'Practice pausing after key points to let them sink in.';
    if (score >= 40) return 'Add more strategic pauses - aim for 3-5 per minute.';
    return 'Start using pauses instead of filler words like "um" and "uh".';
  };

  const getHealthImprovementTip = (score) => {
    if (score >= 90) return 'Excellent vocal health! Your voice is clear and well-supported.';
    if (score >= 80) return 'Minor vocal optimizations could get you to outstanding level.';
    if (score >= 60) return 'Consider vocal coaching to improve clarity and reduce breathiness.';
    if (score >= 40) return 'Practice proper breathing and vocal warm-ups before speaking.';
    return 'Focus on diaphragmatic breathing and consider seeing a speech therapist.';
  };

  const getReadinessImprovementTip = (score) => {
    if (score >= 90) return 'You\'re ready for any professional speaking opportunity!';
    if (score >= 80) return 'Polish a few areas to reach outstanding professional level.';
    if (score >= 60) return 'You\'re well-prepared for most presentations. Focus on weaker areas.';
    if (score >= 40) return 'Good foundation. Practice regularly to reach advanced level.';
    return 'Focus on fundamentals: breathing, pacing, and vocal variety.';
  };

  // Listen for data from video report page
  useEffect(() => {
    const handleMessage = (event) => {
      // Verify origin for security
      if (event.origin !== window.location.origin) {
        return;
      }

      console.log('[AudioReport] Received message:', event.data);

      if (event.data.type === 'AUDIO_REPORT_DATA') {
        console.log('[AudioReport] Received final data from video report:', event.data.data);
        setReportData(event.data.data);
        setDataSource('video');
        setIsLoading(false);
        setLoadingProgress(100);
        setLoadingMessage('Analysis complete!');
      } else if (event.data.type === 'AUDIO_ANALYSIS_PROGRESS') {
        console.log('[AudioReport] Received progress update:', event.data.data);
        setLoadingProgress(event.data.data.progress || 0);
        setLoadingMessage(event.data.data.message || 'Processing...');
      } else if (event.data.type === 'AUDIO_ANALYSIS_ERROR') {
        console.error('[AudioReport] Received analysis error:', event.data.error);
        setAnalysisError(event.data.error);
        setIsLoading(false);
      }
    };

    window.addEventListener('message', handleMessage);

    // If we don't have data and we're not in loading state, we came from direct upload
    if (location.state?.reportData) {
      setDataSource('direct');
      setIsLoading(false);
    }

    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, [location.state]);

  if (analysisError) {
    return (
      <div style={styles.container}>
        <div style={styles.noDataMessage}>
          <h1>Analysis Failed</h1>
          <p style={{ color: '#dc3545', marginTop: '1rem' }}>
            {analysisError}
          </p>
          <button
            onClick={() => window.close()}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer'
            }}
          >
            Close Window
          </button>
        </div>
      </div>
    );
  }

  if (isLoading || !reportData) {
    return (
      <div style={styles.container}>
        <div style={styles.reportBoxOuter}>
          <div style={styles.reportBoxInner}>
            <div style={styles.header}>
              <h1 style={styles.title}>Processing Audio Analysis</h1>
              <p style={styles.subtitle}>
                {dataSource === 'video'
                  ? 'Extracting and analyzing audio from your video...'
                  : 'Please wait while we process your audio data...'
                }
              </p>

              {/* Loading Progress */}
              <div style={{
                maxWidth: '600px',
                margin: '2rem auto',
                padding: '2rem',
                backgroundColor: 'white',
                borderRadius: '1.5rem',
                boxShadow: '0 6px 20px rgba(0,0,0,0.1)'
              }}>
                <div style={{
                  fontSize: '1.2rem',
                  color: '#5D2E8C',
                  marginBottom: '1rem',
                  textAlign: 'center'
                }}>
                  {loadingMessage}
                </div>

                <div style={{
                  width: '100%',
                  height: '1.2rem',
      backgroundColor: 'rgba(255,255,255,0.3)',
                  borderRadius: '0.6rem',
                  overflow: 'hidden',
                  marginBottom: '1rem'
                }}>
<div style={{ ...styles.progressBar, width: `${loadingProgress}%`,  backgroundImage:'repeating-linear-gradient(135deg, #B288C0 0 20px, #ffffff 10px 30px)' }} />
                </div>

                <div style={{
                  textAlign: 'center',
                  fontSize: '1rem',
                  color: '#666',
                  fontWeight: '600'
                }}>
                  {loadingProgress}%
                </div>

                {dataSource === 'video' && (
                  <div style={{
                    marginTop: '1rem',
                    fontSize: '0.9rem',
                    color: '#666',
                    textAlign: 'center',
                    fontStyle: 'italic'
                  }}>
                    This may take a few minutes for detailed vocal analysis...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Safe helper functions
  const safeString = (value, defaultValue = '') => {
    if (value === null || value === undefined) return defaultValue;
    return String(value);
  };

  const safeNumber = (value, defaultValue = 0) => {
    if (value === null || value === undefined || isNaN(value)) return defaultValue;
    return Number(value);
  };

  // Safe number with clamping to 0-100 range
  const safeScore = (value, defaultValue = 0) => {
    if (value === null || value === undefined || isNaN(value)) return defaultValue;
    const num = Number(value);
    return Math.max(0, Math.min(100, num)); // Clamp between 0-100
  };

  const safeArray = (value, defaultValue = []) => {
    if (!Array.isArray(value)) return defaultValue;
    return value;
  };

  const safeObject = (value, defaultValue = {}) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return defaultValue;
    return value;
  };

  if (!reportData) {
    return (
      <div style={styles.container}>
        <div style={styles.noDataMessage}>
          <h1>Loading audio analysis...</h1>
          <p>Please wait while we process your audio data.</p>
          {dataSource === 'video' && (
            <p style={{ marginTop: '1rem', color: '#666' }}>
              Processing audio extracted from your video analysis...
            </p>
          )}
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
  const overallScore = safeScore(reportData.overallScore, 0);

  // Safe nested object access
  const presentationMetrics = safeObject(reportData.presentationMetrics);
  const enhancement = safeObject(reportData.enhancement);
  const speechAnalysis = safeObject(reportData.speechAnalysis);

  // Individual metric scores with safe access (clamped to 0-100)
  const clarityScore = safeScore(presentationMetrics.clarity_score, 0);
  const paceScore = safeScore(presentationMetrics.pace_score, 0);
  const confidenceScore = safeScore(presentationMetrics.confidence_score, 0);
  const engagementScore = safeScore(presentationMetrics.engagement_score, 0);

  // Safe feedback strings
  const clarityFeedback = safeString(presentationMetrics.clarity_feedback, 'Focus on making your message clear and well-structured.');
  const paceFeedback = safeString(presentationMetrics.pace_feedback || speechAnalysis.pace_feedback, 'Work on maintaining an appropriate speaking pace.');
  const confidenceFeedback = safeString(presentationMetrics.confidence_feedback || speechAnalysis.filler_feedback, 'Focus on reducing filler words and speaking with conviction.');
  const engagementFeedback = safeString(presentationMetrics.engagement_feedback, 'Add more energy and enthusiasm to capture audience attention.');

  // Safe arrays
  const keyChanges = safeArray(enhancement.key_changes);
  const speakingTips = safeArray(enhancement.speaking_tips);

  // Safe speech analysis values
  //const wordCount = safeNumber(speechAnalysis.word_count, 0);
  const speakingRate = safeNumber(speechAnalysis.speaking_rate, 0);

  const radarData = [
    { subject: 'Clarity', A: clarityScore, fullMark: 100 },
    { subject: 'Pace', A: paceScore, fullMark: 100 },
    { subject: 'Confidence', A: confidenceScore, fullMark: 100 },
    { subject: 'Engagement', A: engagementScore, fullMark: 100 }
  ];

  // Safe helper functions for display
  const getScoreColor = (score) => {
    const safeScore = Math.max(0, Math.min(100, Number(score) || 0)); // Ensure 0-100 range
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

  const getAdvancedScoreColor = (score) => {
    const safeScoreValue = Math.max(0, Math.min(100, Number(score) || 0)); // Ensure 0-100 range
    if (safeScoreValue >= 90) return '#007bff';      // Outstanding - Blue
    if (safeScoreValue >= 80) return '#28a745';      // Excellent - Green
    if (safeScoreValue >= 60) return '#fd7e14';      // Good - Orange
    if (safeScoreValue >= 40) return '#ffc107';      // Intermediate - Yellow
    return '#dc3545';                               // Beginner - Red
  };

  const getScoreRangeText = (score) => {
    const safeScoreValue = Math.max(0, Math.min(100, Number(score) || 0)); // Ensure 0-100 range
    if (safeScoreValue >= 90) return 'Outstanding';
    if (safeScoreValue >= 80) return 'Excellent';
    if (safeScoreValue >= 60) return 'Good';
    if (safeScoreValue >= 40) return 'Intermediate';
    return 'Beginner';
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
      link.href = `${BASE_URL}/${enhancedAudioUrl}`;
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
      {/* Add responsive CSS */}
      <style>
        {`
          @media (max-width: 768px) {
            .score-section {
              grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)) !important;
              gap: 1rem !important;
            }
            .score-card {
              padding: 1rem !important;
            }
            .score-label {
              font-size: 0.9rem !important;
              line-height: 1.1 !important;
            }
          }
          
          @media (max-width: 480px) {
            .score-section {
              grid-template-columns: repeat(2, 1fr) !important;
            }
          }
          
          .text-overflow-ellipsis {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }

          @media (max-width: 768px) {
            .report-container {
              grid-template-columns: 1fr;    /* one column */
            }
            /* make sure both panels take full width */
            .report-container > div {
              width: 100% !important;
            }
            .transcript-card {
              grid-template-columns: 1fr;
              flex-direction: column !important;
            }
          }
        `}
      </style>

      <div ref={reportRef} className={isExporting ? 'screenshot-mode' : undefined} style={styles.reportBoxOuter}>
        <div style={styles.reportBoxInner}>
          <div style={styles.header}>
            <h1 style={styles.title}>Here Are Your Results!</h1>
            <div style={styles.divider}></div>

            {/* NEW: Source Indicator
            <div style={styles.sourceIndicator}>
              <p style={styles.sourceText}>
                {dataSource === 'video' ? '🎥 Audio extracted from video analysis' : '🎵 Direct audio analysis'}
              </p>
              <p style={styles.sourceSubtext}>
                {dataSource === 'video'
                  ? 'This analysis was generated from the audio track of your video presentation'
                  : 'This analysis was generated from your uploaded audio file'
                }
              </p>
            </div> */}


            <div className="report-container" style={styles.reportContainer}>

              {/* LEFT PANEL */}
              <div className="left-panel" style={styles.leftPanel}>
                {/* Presentation Scores */}
                <div style={{ textAlign: 'center', justifyContent: 'center', marginBottom: '10px', color: '#5D2E8C', fontSize: '20px', fontWeight: '600' }}>
                  Performance Radar</div>
                <RadarChart width={chartWidth} height={chartHeight} cx={chartWidth / 2} data={radarData} outerRadius={outerRadius}>
                  <PolarGrid stroke="#5D2E8C" />
                  <PolarAngleAxis
                    dataKey="subject"
                    stroke="#5D2E8C"
                    tick={{ fontSize: 12, fill: '#5D2E8C' }}
                    className="radar-labels"
                  />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Score" dataKey="A" stroke="#5D2E8C" fill="#E2779C" fillOpacity={0.5} />
                  <Tooltip />
                </RadarChart>
              </div>

              {/* RIGHT PANEL */}
              <div className="right-panel" style={styles.rightPanel}>
                {/* Meta Information */}
                <div style={styles.metaItem}>
                  <div style={styles.metaLabel}>Score</div>
                  <div style={{ ...styles.metaValue, color: getScoreColor(overallScore) }}>{overallScore}</div>
                </div>
                <div style={styles.metaItem}>
                  <div style={styles.metaLabel}>Language</div>
                  <div style={styles.metaValue}>{safeLanguage}</div>
                </div>
                <div style={styles.metaItem}>
                  <div style={styles.metaLabel}>Speaking Rate</div>
                  <div style={styles.metaValue}>{speakingRate} WPM</div>
                </div>
              </div>
            </div>

          <div className="report-container" style={{...styles.reportContainer, paddingTop: '20px'}}>
              <div className="left-panel" style={styles.leftPanel}>
                <div style={styles.metaItem}>
                  <div style={styles.metaLabel}>Duration</div>
                  <div style={styles.metaValue}>{formatDuration(duration)}</div>
                </div>
              </div>
              
              <div className="right-panel" style={styles.rightPanel}>
                {/* Meta Information */}
                <div style={{ ...styles.metaItem, ...styles.userInfoCard }}>
                  <div style={styles.metaLabel}>Username</div>
                  <div style={styles.userInfoValue}>{username}</div>
                  <div style={styles.metaLabel}>Date</div>
                  <div style={styles.userInfoValue}>{date}</div>
                </div>
              </div>

            </div>
            
          </div>

          {/* End header */}

          {/* Vocal Dynamics Section */}
          {reportData.vocalDynamics && (
            <div style={styles.vocalDynamicsSection}>
              {/* Vocal Dynamics Header */}
              <div style={styles.vocalDynamicsHeader}>
                <h2 style={styles.sectionTitle}>Advanced Vocal Analysis</h2>
                <div style={styles.overallDynamicsScore}>
                  <div style={{
                    ...styles.scoreNumber,
                    color: getScoreColor(safeScore(reportData.vocalDynamics.overall_dynamics_score, 0)),
                    fontSize: '2.5rem'
                  }}>
                    {safeScore(reportData.vocalDynamics.overall_dynamics_score, 0)}
                  </div>
                  <div style={styles.scoreLabel}>Vocal Dynamics Score</div>
                  <div style={styles.readinessLevel}>
                    {safeString(reportData.vocalDynamics.presentation_readiness?.readiness_level, 'Beginner')} Level
                  </div>
                  <div style={styles.scoreRange}>
                    {getScoreRangeText(safeScore(reportData.vocalDynamics.overall_dynamics_score, 0))}
                  </div>
                </div>
              </div>

              {/* Score Range Legend */}
              <div style={styles.scoreRangeLegend}>
                <h3 style={styles.legendTitle}>Performance Levels</h3>
                <div style={styles.legendGrid}>
                  <div style={styles.legendItem}>
                    <div style={{ ...styles.legendColor, backgroundColor: '#dc3545' }}></div>
                    <span style={styles.legendText}>Beginner (0-39)</span>
                  </div>
                  <div style={styles.legendItem}>
                    <div style={{ ...styles.legendColor, backgroundColor: '#ffc107' }}></div>
                    <span style={styles.legendText}>Intermediate (40-59)</span>
                  </div>
                  <div style={styles.legendItem}>
                    <div style={{ ...styles.legendColor, backgroundColor: '#fd7e14' }}></div>
                    <span style={styles.legendText}>Good (60-79)</span>
                  </div>
                  <div style={styles.legendItem}>
                    <div style={{ ...styles.legendColor, backgroundColor: '#28a745' }}></div>
                    <span style={styles.legendText}>Excellent (80-89)</span>
                  </div>
                  <div style={styles.legendItem}>
                    <div style={{ ...styles.legendColor, backgroundColor: '#007bff' }}></div>
                    <span style={styles.legendText}>Outstanding (90-100)</span>
                  </div>
                </div>
              </div>

              {/* Vocal Dynamics Grid */}
              <div style={styles.vocalDynamicsGrid}>

                {/* Pitch Dynamics Card */}
                <div style={styles.vocalCard}>
                  <div style={styles.vocalCardHeader}>
                    <div style={styles.iconContainer}>
                      <svg style={styles.cardIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    </div>
                    <div>
                      <h3 style={styles.vocalCardTitle}>Pitch Variation</h3>
                      <div style={{
                        ...styles.vocalScore,
                        color: getAdvancedScoreColor(safeScore(reportData.vocalDynamics.pitch_dynamics?.variation_score, 0))
                      }}>
                        {safeScore(reportData.vocalDynamics.pitch_dynamics?.variation_score, 0)}/100
                      </div>
                      <div style={styles.scoreRangeIndicator}>
                        {getScoreRangeText(safeScore(reportData.vocalDynamics.pitch_dynamics?.variation_score, 0))}
                      </div>
                    </div>
                  </div>
                  <div style={styles.vocalCardContent}>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Dynamic Range:</span>
                      <span style={styles.metricValue}>
                        {safeString(reportData.vocalDynamics.pitch_dynamics?.dynamic_range, 'Moderate')}
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Monotone Risk:</span>
                      <span style={{
                        ...styles.metricValue,
                        color: safeScore(reportData.vocalDynamics.pitch_dynamics?.monotone_risk, 50) > 70 ? '#dc3545' :
                          safeScore(reportData.vocalDynamics.pitch_dynamics?.monotone_risk, 50) > 40 ? '#ffc107' : '#28a745'
                      }}>
                        {safeScore(reportData.vocalDynamics.pitch_dynamics?.monotone_risk, 50)}%
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Pitch Range:</span>
                      <span style={styles.metricValue}>
                        {safeNumber(reportData.vocalDynamics.pitch_dynamics?.pitch_range_hz, 0).toFixed(1)} Hz
                      </span>
                    </div>
                    <div style={styles.improvementTip}>
                      <strong>Next Level:</strong> {getPitchImprovementTip(safeScore(reportData.vocalDynamics.pitch_dynamics?.variation_score, 0))}
                    </div>
                  </div>
                </div>

                {/* Volume Dynamics Card */}
                <div style={styles.vocalCard}>
                  <div style={styles.vocalCardHeader}>
                    <div style={styles.iconContainer}>
                      <svg style={styles.cardIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 12.536a4 4 0 010-5.656m0 0a4 4 0 010 5.656m-2.828-2.828a2.5 2.5 0 010-3.536" />
                      </svg>
                    </div>
                    <div>
                      <h3 style={styles.vocalCardTitle}>Volume Dynamics</h3>
                      <div style={{
                        ...styles.vocalScore,
                        color: getAdvancedScoreColor(safeScore(reportData.vocalDynamics.volume_dynamics?.energy_score, 0))
                      }}>
                        {safeScore(reportData.vocalDynamics.volume_dynamics?.energy_score, 0)}/100
                      </div>
                      <div style={styles.scoreRangeIndicator}>
                        {getScoreRangeText(safeScore(reportData.vocalDynamics.volume_dynamics?.energy_score, 0))}
                      </div>
                    </div>
                  </div>
                  <div style={styles.vocalCardContent}>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Energy Presence:</span>
                      <span style={styles.metricValue}>
                        {safeString(reportData.vocalDynamics.volume_dynamics?.dynamic_presence, 'Moderate')}
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Volume Range:</span>
                      <span style={styles.metricValue}>
                        {safeNumber(reportData.vocalDynamics.volume_dynamics?.volume_range, 0).toFixed(1)} dB
                      </span>
                    </div>
                    <div style={styles.improvementTip}>
                      <strong>Next Level:</strong> {getVolumeImprovementTip(safeScore(reportData.vocalDynamics.volume_dynamics?.energy_score, 0))}
                    </div>
                  </div>
                </div>

                {/* Rhythm Analysis Card */}
                <div style={styles.vocalCard}>
                  <div style={styles.vocalCardHeader}>
                    <div style={styles.iconContainer}>
                      <svg style={styles.cardIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h3 style={styles.vocalCardTitle}>Speaking Rhythm</h3>
                      <div style={{
                        ...styles.vocalScore,
                        color: getAdvancedScoreColor(safeScore(reportData.vocalDynamics.rhythm_analysis?.rhythm_score, 0))
                      }}>
                        {safeScore(reportData.vocalDynamics.rhythm_analysis?.rhythm_score, 0)}/100
                      </div>
                      <div style={styles.scoreRangeIndicator}>
                        {getScoreRangeText(safeScore(reportData.vocalDynamics.rhythm_analysis?.rhythm_score, 0))}
                      </div>
                    </div>
                  </div>
                  <div style={styles.vocalCardContent}>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Pace Category:</span>
                      <span style={styles.metricValue}>
                        {safeString(reportData.vocalDynamics.rhythm_analysis?.pace_category, 'Moderate')}
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Estimated Pace:</span>
                      <span style={styles.metricValue}>
                        {Math.max(80, Math.min(200, safeNumber(reportData.vocalDynamics.rhythm_analysis?.estimated_pace_wpm, 150)))} WPM
                      </span>
                    </div>
                    <div style={styles.improvementTip}>
                      <strong>Next Level:</strong> {getRhythmImprovementTip(safeScore(reportData.vocalDynamics.rhythm_analysis?.rhythm_score, 0))}
                    </div>
                  </div>
                </div>

                {/* Pause Analysis Card */}
                <div style={styles.vocalCard}>
                  <div style={styles.vocalCardHeader}>
                    <div style={styles.iconContainer}>
                      <svg style={styles.cardIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h3 style={styles.vocalCardTitle}>Strategic Pauses</h3>
                      <div style={{
                        ...styles.vocalScore,
                        color: getAdvancedScoreColor(safeScore(reportData.vocalDynamics.pause_analysis?.strategic_score, 0))
                      }}>
                        {safeScore(reportData.vocalDynamics.pause_analysis?.strategic_score, 0)}/100
                      </div>
                      <div style={styles.scoreRangeIndicator}>
                        {getScoreRangeText(safeScore(reportData.vocalDynamics.pause_analysis?.strategic_score, 0))}
                      </div>
                    </div>
                  </div>
                  <div style={styles.vocalCardContent}>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Pause Effectiveness:</span>
                      <span style={styles.metricValue}>
                        {safeString(reportData.vocalDynamics.pause_analysis?.pause_effectiveness, 'Moderate')}
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Average Pause:</span>
                      <span style={styles.metricValue}>
                        {Math.max(0.1, Math.min(5.0, safeNumber(reportData.vocalDynamics.pause_analysis?.avg_pause_duration, 0.5))).toFixed(1)}s
                      </span>
                    </div>
                    <div style={styles.improvementTip}>
                      <strong>Next Level:</strong> {getPauseImprovementTip(safeScore(reportData.vocalDynamics.pause_analysis?.strategic_score, 0))}
                    </div>
                  </div>
                </div>

                {/* Vocal Health Card */}
                <div style={styles.vocalCard}>
                  <div style={styles.vocalCardHeader}>
                    <div style={styles.iconContainer}>
                      <svg style={styles.cardIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                    </div>
                    <div>
                      <h3 style={styles.vocalCardTitle}>Vocal Health</h3>
                      <div style={{
                        ...styles.vocalScore,
                        color: getAdvancedScoreColor(safeScore(reportData.vocalDynamics.vocal_health_indicators?.vocal_clarity, 0))
                      }}>
                        {safeScore(reportData.vocalDynamics.vocal_health_indicators?.vocal_clarity, 0)}/100
                      </div>
                      <div style={styles.scoreRangeIndicator}>
                        {getScoreRangeText(safeScore(reportData.vocalDynamics.vocal_health_indicators?.vocal_clarity, 0))}
                      </div>
                    </div>
                  </div>
                  <div style={styles.vocalCardContent}>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Breathiness Level:</span>
                      <span style={styles.metricValue}>
                        {safeScore(reportData.vocalDynamics.vocal_health_indicators?.breathiness_level, 25)}%
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Voice Stability:</span>
                      <span style={styles.metricValue}>
                        {safeString(reportData.vocalDynamics.vocal_health_indicators?.voice_stability, 'Good')}
                      </span>
                    </div>
                    <div style={styles.improvementTip}>
                      <strong>Next Level:</strong> {getHealthImprovementTip(safeScore(reportData.vocalDynamics.vocal_health_indicators?.vocal_clarity, 0))}
                    </div>
                  </div>
                </div>

                {/* Overall Readiness Card */}
                <div style={styles.vocalCard}>
                  <div style={styles.vocalCardHeader}>
                    <div style={styles.iconContainer}>
                      <svg style={styles.cardIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                      </svg>
                    </div>
                    <div>
                      <h3 style={styles.vocalCardTitle}>Presentation Readiness</h3>
                      <div style={{
                        ...styles.vocalScore,
                        color: getAdvancedScoreColor(safeScore(reportData.vocalDynamics.overall_dynamics_score, 0))
                      }}>
                        {safeScore(reportData.vocalDynamics.overall_dynamics_score, 0)}/100
                      </div>
                      <div style={styles.scoreRangeIndicator}>
                        {safeString(reportData.vocalDynamics.presentation_readiness?.readiness_level, 'Beginner')}
                      </div>
                    </div>
                  </div>
                  <div style={styles.vocalCardContent}>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Professional Level:</span>
                      <span style={styles.metricValue}>
                        {safeString(reportData.vocalDynamics.presentation_readiness?.professional_level, 'Developing')}
                      </span>
                    </div>
                    <div style={styles.vocalMetric}>
                      <span style={styles.metricLabel}>Improvement Areas:</span>
                      <span style={styles.metricValue}>
                        {Math.max(0, Math.min(6, safeNumber(reportData.vocalDynamics.presentation_readiness?.improvement_areas_count, 2)))}
                      </span>
                    </div>
                    <div style={styles.improvementTip}>
                      <strong>Next Level:</strong> {getReadinessImprovementTip(safeScore(reportData.vocalDynamics.overall_dynamics_score, 0))}
                    </div>
                  </div>
                </div>

              </div>

              {/* Professional Benchmark Comparison */}
              <div style={styles.benchmarkCard}>
                <h3 style={styles.cardTitle}>Professional Speaker Comparison</h3>
                <div style={styles.benchmarkContent}>
                  <div style={styles.benchmarkItem}>
                    <span style={styles.benchmarkLabel}>Your Overall Score:</span>
                    <span style={styles.benchmarkValue}>
                      {safeScore(reportData.vocalDynamics.overall_dynamics_score, 0)}/100
                    </span>
                  </div>
                  <div style={styles.benchmarkItem}>
                    <span style={styles.benchmarkLabel}>Professional Benchmark:</span>
                    <span style={styles.benchmarkValue}>85+</span>
                  </div>
                  <div style={styles.benchmarkItem}>
                    <span style={styles.benchmarkLabel}>TED Talk Average:</span>
                    <span style={styles.benchmarkValue}>78</span>
                  </div>
                  <div style={styles.benchmarkItem}>
                    <span style={styles.benchmarkLabel}>Corporate Presentation Average:</span>
                    <span style={styles.benchmarkValue}>65</span>
                  </div>
                </div>
              </div>

              {/* Vocal Summary */}
              <div style={styles.vocalSummaryCard}>
                <h3 style={styles.cardTitle}>Vocal Analysis Summary</h3>
                <div style={styles.vocalSummaryContent}>
                  {safeString(reportData.speechAnalysis?.vocal_summary, 'Vocal analysis summary not available.')}
                </div>
              </div>
            </div>
          )}

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
                  <h4 style={{ color: '#2d5a2d', marginBottom: '1rem', fontSize: '1.1rem' }}>
                    Listen to Your Enhanced Speech
                  </h4>
                  <div style={styles.audioControls}>
                    <button
                      style={{ ...styles.playButton, ...styles.playButtonEnhanced }}
                      onClick={handlePlayEnhanced}
                    >
                      {isPlayingEnhanced ? '⏸️ Pause Enhanced' : 'Play Enhanced Speech'}
                    </button>

                    <button
                      style={{ ...styles.playButton, ...styles.downloadButton }}
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
                    style={{ width: '100%', marginTop: '1rem' }}
                    onPlay={() => setIsPlayingEnhanced(true)}
                    onPause={() => setIsPlayingEnhanced(false)}
                    onEnded={() => setIsPlayingEnhanced(false)}
                    onError={() => setAudioError('Audio file could not be loaded. Please try again later.')}
                  >
                    <source src={`${BASE_URL}/${enhancedAudioUrl}`} type="audio/mpeg" />
                    Your browser does not support the audio element.
                  </audio>
                </div>
              )}

              {/* Key Changes Made */}
              {keyChanges.length > 0 && (
                <div style={styles.improvementsList}>
                  <h4 style={{ margin: '0 0 1rem 0', color: '#5D2E8C' }}>Key Improvements Made:</h4>
                  {keyChanges.map((change, index) => (
                    <div key={index} style={styles.improvementItem}>
                      <span style={{ color: '#28a745', fontSize: '1.2rem', fontWeight: 'bold' }}>✓</span>
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
              <h3 style={{ color: '#856404', marginBottom: '1rem', fontSize: '1.4rem' }}>
                Personalised Speaking Tips
              </h3>
              {speakingTips.map((tip, index) => (
                <div key={index} style={styles.tipItem}>
                  <span style={{ fontSize: '1.2rem' }}>💡 </span>
                  <span>{safeString(tip, 'Speaking tip')}</span>
                </div>
              ))}
            </div>
          )}

          {/* View Toggle */}
          {enhancedTranscript && renderViewToggle()}

          {/* Transcript Comparison */}
          <div class="transcript-card" style={styles.transcriptGrid}>
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
                <span style={styles.metricIcon}>🎯</span>
                <span style={styles.metricTitle}>Clarity Analysis</span>
              </div>
              <div style={styles.feedback}>{clarityFeedback}</div>
            </div>

            <div style={styles.metricCard}>
              <div style={styles.metricHeader}>
                <span style={styles.metricIcon}>⏱️</span>
                <span style={styles.metricTitle}>Pace Analysis</span>
              </div>
              <div style={styles.feedback}>{paceFeedback}</div>
            </div>

            <div style={styles.metricCard}>
              <div style={styles.metricHeader}>
                <span style={styles.metricIcon}>💪</span>
                <span style={styles.metricTitle}>Confidence Analysis</span>
              </div>
              <div style={styles.feedback}>{confidenceFeedback}</div>
            </div>

            <div style={styles.metricCard}>
              <div style={styles.metricHeader}>
                <span style={styles.metricIcon}>✨</span>
                <span style={styles.metricTitle}>Engagement Analysis</span>
              </div>
              <div style={styles.feedback}>{engagementFeedback}</div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
        <button className="download-button" onClick={handlePrint}>
<FaDownload/>{isExporting ? 'Generating…' : 'Download Report'}
        </button>
      </div>

      <style>{`

.download-button {
  background-color: #5D2E8C;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  justify-content: center;
  gap: 8px;
  transition: background-color 0.3s, transform 0.1s;
}

.download-button:hover {
  background-color: #5a3fa0; /* slightly darker purple on hover */
}

.download-button:active {
  transform: scale(0.95); /* slightly shrink on click */
}

@media (max-width: 768px) {
  h1 { font-size: 22px !important; }
  .recharts-polar-angle-axis-tick-value { font-size: 9px !important; }
}
.screenshot-mode,
        .screenshot-mode * {       /* flatten BG */
          box-shadow: none !important;        /* remove shadows */
          text-shadow: none !important;
          filter: none !important;
          transition: none !important;
          animation: none !important;
        }
        .screenshot-mode .scroll-chart {
          overflow: hidden !important;        /* hide scrollbars */
        }
`}</style>
    </div>

  );
}

