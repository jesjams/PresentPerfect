import { useLocation, useNavigate } from 'react-router-dom';
import { ref, uploadString, getDownloadURL } from 'firebase/storage';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip
} from 'recharts';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { PieChart, Pie, Cell, Legend } from 'recharts';
import { ReactComponent as Mansvg } from '../assets/mansvgrepo.svg';
import html2canvas from 'html2canvas';
import { useAuth } from '../context/AuthContext';
import { FaDownload, FaMicrophone } from 'react-icons/fa';
import { useRef, useState } from 'react';
import QRCode from 'react-qr-code';
import { Tooltip as RTTooltip } from 'react-tooltip';
import 'react-tooltip/dist/react-tooltip.css';
import { storage } from '../context/firebase';
import { v4 as uuid } from 'uuid';
const BASE_URL = process.env.REACT_APP_API_HOST;
const SOCKET_HOST = process.env.REACT_APP_SOCKET_HOST || 'http://localhost:4000';

export default function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();

  const reportData = location.state?.reportData;
  const { user } = useAuth();
  const username = user?.email?.split('@')[0] || 'Guest';

  const date = new Date().toLocaleDateString('en-AU', {
    weekday: 'short',
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
  const [isExporting, setIsExporting] = useState(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const [modalOpen,   setModalOpen]     = useState(false);
   const [imgDataUrl,  setImgDataUrl]    = useState(''); 
   const [imgUrl,  setImgUrl]    = useState(''); 
  const reportRef = useRef(null);
const handlePrint = async () => {
  setIsExporting(true);

  await new Promise(r => setTimeout(r, 50));

  try {
    const canvas = await html2canvas(reportRef.current, {
      scale: 1,
      useCORS: true,
      width: 1050,
      windowWidth: 1100
    });

    const dataUrl = canvas.toDataURL('image/png');

    // ✅ Correct way to create reference and upload
    const fileName = `reports/${uuid()}.png`;
    const fileRef  = ref(storage, fileName); // ref comes from firebase/storage
    setImgUrl(dataUrl);
    await uploadString(fileRef, dataUrl, 'data_url');

    // ✅ Get downloadable URL
    const downloadUrl = await getDownloadURL(fileRef);

    setImgDataUrl(downloadUrl);
    setModalOpen(true);
  } catch (err) {
    console.error('Upload error:', err);
  } finally {
    setIsExporting(false);
  }
};


  /* --- download from the modal --- */
  const downloadNow = () => {
    const link = document.createElement('a');
    link.href     = imgUrl;
    link.download = 'Performance_Report.png';
    link.click();
  };

  const handleAudioAnalysis = async () => {
    setIsProcessingAudio(true);

    // Declare audioReportWindow outside try block to fix scope issue
    let audioReportWindow = null;

    try {
      // Generate a unique report ID for this session
      const audioReportId = `audio_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Open audio report in new tab immediately
      audioReportWindow = window.open('/audio-report', '_blank');

      if (!audioReportWindow) {
        alert('Please allow popups for this site to open the audio analysis in a new tab.');
        setIsProcessingAudio(false);
        return;
      }

      // We'll use the video file that was processed for the current report
      // In a real implementation, you'd store the video file path in the backend
      // For now, we'll use a placeholder approach where the backend knows which video to use
      const payload = {
        reportId: audioReportId,
        videoReportId: reportData.reportId || Date.now(),
        // Note: In production, you'd have the backend track the video file path
        // associated with each report ID rather than passing paths from frontend
      };

      // Call the new API endpoint to extract audio and analyze
      const response = await fetch(`${BASE_URL}/api/extract-audio-and-analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === 'success') {
        console.log('[VideoReport] Audio extraction started successfully');

        // Set up real-time communication with the audio report window
        const messageHandler = (event) => {
          if (event.origin !== window.location.origin) return;

          // Listen for progress updates and pass them to the audio report window
          if (event.data.type?.startsWith('video-audio-analysis')) {
            audioReportWindow.postMessage(event.data, window.location.origin);
          }
        };

        window.addEventListener('message', messageHandler);

        // Setup socket listeners for this specific analysis
        import('socket.io-client').then(({ io }) => {
          const socket = io(SOCKET_HOST);

          socket.on('video-audio-analysis-update', (data) => {
            if (data.reportId === audioReportId) {
              audioReportWindow.postMessage({
                type: 'AUDIO_ANALYSIS_PROGRESS',
                data: data
              }, window.location.origin);
            }
          });

          socket.on('video-audio-analysis-complete', (data) => {
            if (data.reportId === audioReportId) {
              audioReportWindow.postMessage({
                type: 'AUDIO_REPORT_DATA',
                data: data.data
              }, window.location.origin);
              socket.disconnect();
              window.removeEventListener('message', messageHandler);
            }
          });

          socket.on('video-audio-analysis-error', (data) => {
            if (data.reportId === audioReportId) {
              audioReportWindow.postMessage({
                type: 'AUDIO_ANALYSIS_ERROR',
                error: data.error
              }, window.location.origin);
              socket.disconnect();
              window.removeEventListener('message', messageHandler);
            }
          });
        });

      } else {
        console.error('Audio analysis failed:', result.error);
        alert('Audio analysis failed: ' + result.error);
        if (audioReportWindow) {
          audioReportWindow.close();
        }
      }
    } catch (error) {
      console.error('Error during audio analysis:', error);
      alert('Failed to start audio analysis. Please try again.');
      if (audioReportWindow && !audioReportWindow.closed) {
        audioReportWindow.close();
      }
    } finally {
      setIsProcessingAudio(false);
    }
  };

  if (!reportData) {
    navigate('/');
    return null;
  }

  const {
    emotionScore,
    gazeScore,
    movementScore,
    shoulderScore,
    handsScore,
    speechScore,
    overallScore,
    overallSummary
  } = reportData;

  const maxSecond = Math.max(...Object.keys(reportData.emotion).map(Number));

  const { emotionBySegment } = reportData;
  const emotionPerSegment = emotionBySegment;

  const segments = reportData.transcriptSegments
    .split("\n")
    .filter(Boolean)
    .map(line => {
      const m = line.match(/^\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s]\s*(.*)/);
      if (!m) return null;
      return {
        start: parseFloat(m[1]),
        end: parseFloat(m[2]),
        text: m[3],
      };
    })
    .filter(Boolean);

  const totalDur = reportData.videoDuration || (segments.at(-1)?.end ?? 1);


  const emotionColors = {
    Neutral: '#cccccc',
    Happy: '#ffd700',
    Sad: '#1e90ff',
    Surprise: '#ff6347',
    Anger: '#ff0400',
  };

  const emoSeg = emotionBySegment.length === segments.length
    ? emotionBySegment
    : Array(segments.length).fill("None");


  const gazePerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.gaze[sec] || 'None'
  );
  const gazeCounts = gazePerSecond.reduce((acc, val) => {
    const cleaned = val.trim().toLowerCase();
    acc[cleaned] = (acc[cleaned] || 0) + 1;
    return acc;
  }, {});
  const total = gazePerSecond.length;
  const getHeatColor = (percent) => {
    const baseAlpha = 0.1;
    const extraAlpha = Math.min(0.7, percent / 100);
    return `rgba(107, 76, 175, ${baseAlpha + extraAlpha})`;
  };
  const gazePercentages = {
    'up left': ((gazeCounts['up-left'] || 0) / total * 100).toFixed(1),
    'up': ((gazeCounts['up'] || 0) / total * 100).toFixed(1),
    'up right': ((gazeCounts['up-right'] || 0) / total * 100).toFixed(1),
    'left': ((gazeCounts['left'] || 0) / total * 100).toFixed(1),
    'center': ((gazeCounts['straight'] || 0) / total * 100).toFixed(1),
    'right': ((gazeCounts['right'] || 0) / total * 100).toFixed(1),
    'down left': ((gazeCounts['down-left'] || 0) / total * 100).toFixed(1),
    'down': ((gazeCounts['down'] || 0) / total * 100).toFixed(1),
    'down right': ((gazeCounts['down-right'] || 0) / total * 100).toFixed(1)
  };

  const shoulderPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.shoulder[sec] || 'None'
  );

  const handsPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.gesture[sec] || 'None'
  );
  const toPercentageData = (arr) => {
    const counts = arr.reduce((acc, val) => {
      const cleaned = val.replace('°', '').trim();
      acc[cleaned] = (acc[cleaned] || 0) + 1;
      return acc;
    }, {});
    const total = arr.length;
    return Object.entries(counts).map(([name, count]) => ({
      name,
      value: Number(((count / total) * 100).toFixed(1))
    }));
  };

  const shoulderData = toPercentageData(shoulderPerSecond);
  const handsData = toPercentageData(handsPerSecond);

  const pieColors = ['#5D2E8C', '#E2779C'];


  const movementPerSecond = Array.from({ length: maxSecond + 1 }, (_, sec) =>
    reportData.movement[sec] !== undefined ? reportData.movement[sec] : 0
  );

  const movementData = movementPerSecond.map((pos, index) => ({
    second: index,
    position: pos,
    label:
      pos <= 2
        ? 'Left'
        : pos <= 4
          ? 'Middle Left'
          : pos <= 6
            ? 'Center'
            : pos <= 8
              ? 'Middle Right'
              : 'Right'
  }));

  let rank = 'E';
  if (overallScore >= 90) rank = 'S';
  else if (overallScore >= 80) rank = 'A';
  else if (overallScore >= 70) rank = 'B';
  else if (overallScore >= 50) rank = 'C';
  else if (overallScore >= 30) rank = 'D';

  const radarData = [
    { subject: 'Emotion', A: emotionScore, fullMark: 100 },
    { subject: 'Gaze', A: gazeScore, fullMark: 100 },
    { subject: 'Movement', A: movementScore, fullMark: 100 },
    { subject: 'Shoulder', A: shoulderScore, fullMark: 100 },
    { subject: 'Gesture', A: handsScore, fullMark: 100 },
    { subject: 'Speech', A: speechScore, fullMark: 100 }
  ];

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
      minWidth: '200px',
      margin: '0 auto'
    },
    reportBoxInner: {
      border: '2px dotted white',
      borderRadius: '15px',
      padding: '20px'
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
    reportContainer: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '20px',
      justifyContent: 'center'
    },
    leftPanel: {
      backgroundColor: '#fff',
      borderRadius: '15px',
      padding: '40px',
      minWidth: '50px',
      justifyContent: 'center',
      justifyItems: 'center',
      flex: '1 1 400px'
    },
    rightPanel: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      flex: '1 1 300px'
    },
    card: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      borderRadius: '15px',
      padding: '20px',
      textAlign: 'center'
    },
    cardTitle: {
      fontSize: '12px',
      textTransform: 'uppercase',
      fontWeight: '500'
    },
    cardValue: {
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
    summarySection: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      borderRadius: '15px',
      padding: '20px',
      textAlign: 'start',
      fontSize: '18px',
      fontWeight: '500',
      lineHeight: '1.5',
      marginTop: '20px',
            marginBottom: '20px',
      wordWrap: 'break-word',
      overflowWrap: 'break-word'
    },
    textSection: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      borderRadius: '15px',
      padding: '0px',
      textAlign: 'start',
      fontSize: '18px',
      fontWeight: '500',
      lineHeight: '1.5',
      marginTop: '0px',
      wordWrap: 'break-word',
      overflowWrap: 'break-word'
    },
    summaryTitle: {
      fontSize: '22px',
      fontWeight: '700',
      marginBottom: '10px'
    },
    // NEW: Audio Analysis Button Styles
    audioAnalysisSection: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      borderRadius: '15px',
      padding: '20px',
      textAlign: 'center',
      marginTop: '20px',
      border: '2px dashed #5D2E8C'
    },
    audioAnalysisTitle: {
      fontSize: '20px',
      fontWeight: '700',
      marginBottom: '10px',
      color: '#5D2E8C'
    },
    audioAnalysisDescription: {
      fontSize: '16px',
      marginBottom: '20px',
      color: '#666',
      lineHeight: '1.4'
    },
    audioAnalysisButton: {
      backgroundColor: '#5D2E8C',
      color: 'white',
      border: 'none',
      padding: '15px 30px',
      borderRadius: '25px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      display: 'inline-flex',
      alignItems: 'center',
      gap: '10px',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 12px rgba(93, 46, 140, 0.3)'
    },
    audioAnalysisButtonDisabled: {
      backgroundColor: '#ccc',
      cursor: 'not-allowed',
      boxShadow: 'none'
    },
    breakdownSection: {
      backgroundColor: 'white',
      color: '#5D2E8C',
      borderRadius: '15px',
      padding: '20px',
      marginTop: '20px',
      textAlign: 'start',
      fontSize: '18px',
      fontWeight: '500',
      lineHeight: '1.5'
    },
    breakdownContent: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
      marginTop: '10px'
    },
    placeholderBox: {
      backgroundColor: '#f1f1f1',
      borderRadius: '10px',
      padding: '15px',
      textAlign: 'center',
    },
    breakdownTitle: {
      fontSize: '26px',
      fontWeight: '700',
      textAlign: 'center',
      marginBottom: '20px'
    },
    graphSection: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      marginBottom: '30px'
    },
    graphTitle: {
      fontSize: '20px',
      fontWeight: '600',
      marginBottom: '10px'
    },
    graphBar: {
      display: 'flex',
      width: '100%',
      height: '60px',
      overflow: 'hidden',
      borderRadius: '10px'
    },
    graphBlockContainer: {
      margin: '0',
      padding: '0'
    },
    graphBlock: {
      flex: '1',
      height: '60px',
      transition: 'background-color 0.3s ease'
    },
    graphBlockLabel: {
      fontSize: '8px',
      color: '#333'
    },
    suggestionsTitle: {
      fontSize: '16px',
      fontWeight: '600',
      marginTop: '10px',
      marginBottom: '5px',
      textAlign: 'left',
    },
    legendContainer: {
      display: 'flex',
      justifyContent: 'center',
      flexWrap: 'wrap',
      marginTop: '10px',
      gap: '10px'
    },
    legendItem: {
      display: 'flex',
      alignItems: 'center',
      fontSize: '12px'
    },
    legendColor: {
      width: '12px',
      height: '12px',
      borderRadius: '2px',
      marginRight: '5px',
      border: '1px solid #ccc'
    },
    legendLabel: {
      fontSize: '12px',
      color: '#333'
    },
    suggestionText: {
      fontSize: '14px',
      color: '#555',
      textAlign: 'left',
      alignSelf: 'flex-aSTART'
    },
    barLabels: {
      display: 'flex',
      justifyContent: 'space-between',
      fontSize: '10px',
      marginTop: '4px',
      padding: '0 4px'
    },
    barLabel: {
      color: '#555'
    },
    backdrop: {
    position: 'fixed', inset: 0,
    background: 'rgba(0,0,0,.45)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    zIndex: 9999
  },
  modal: {
    background: '#fff',
    borderRadius: '1rem',
    padding: '2rem',
    width: 'min(90vw, 400px)',
    textAlign: 'center',
    boxShadow: '0 8px 30px rgba(0,0,0,.2)'
  },
  qrWrap: {
    margin: '0 auto 1rem auto',
    width: 160, height: 160
  },
  dlBtn: {
    padding: '.6rem 1.2rem',
    borderRadius: '.5rem',
    border: 'none',
    fontWeight: 600,
    cursor: 'pointer',
    background: '#5D2E8C', /* keep your brand colour */
    color: '#fff',
    marginRight: '.5rem'
  },
  closeBtn: {
    padding: '.6rem 1.2rem',
    borderRadius: '.5rem',
    border: '1px solid #ccc',
    background: '#fff',
    cursor: 'pointer',
    color: '#333'
  }
  };



  return (
    <div style={styles.container}>
      <div >
        <div ref={reportRef} style={styles.reportBoxOuter}>
          <div style={styles.reportBoxInner}>
            <h1 style={styles.title}>Here Are Your Results!</h1>
            <div style={styles.divider}></div>
                        {/* OVERVIEW SECTION */}
            <div style={styles.summarySection}>
              <div>{overallSummary}</div>
            </div>

            <div className="report-container" style={styles.reportContainer}>
              {/* LEFT PANEL */}
              <div className="left-panel" style={styles.leftPanel}>
                <div style={{ textAlign: 'center', marginBottom: '10px', color: '#5D2E8C', fontSize: '20px', fontWeight: '600' }}>
                  Performance Radar
                </div>
                <RadarChart width={400} height={300} cx={200} cy={150} outerRadius={120} data={radarData}>
                  <PolarGrid stroke="#5D2E8C" />
                  <PolarAngleAxis dataKey="subject" stroke="#5D2E8C" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Score" dataKey="A" stroke="#5D2E8C" fill="#E2779C" fillOpacity={0.5} isAnimationActive={true} />
                  <Tooltip />
                </RadarChart>
              </div>

              {/* RIGHT PANEL */}
              <div className="right-panel" style={styles.rightPanel}>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Score</div>
                  <div style={styles.cardValue}>{overallScore}%</div>
                </div>
                <div style={styles.card}>
                  <div style={styles.cardTitle}>Rank</div>
                  <div style={styles.cardValue}>{rank}</div>
                </div>
                <div style={{ ...styles.card, ...styles.userInfoCard }}>
                  <div style={styles.cardTitle}>Username</div>
                  <div style={styles.userInfoValue}>{username}</div>
                  <div style={styles.cardTitle}>Date</div>
                  <div style={styles.userInfoValue}>{date}</div>
                </div>
              </div>
            </div>

            <div style={styles.divider}></div>
            <h1 style={styles.title}>Speech Analysis</h1>
            {/* TRANSCRIPT & SPEECH IMPROVEMENT SECTIONS */}
            <div style={styles.breakdownSection}>
              <div style={styles.summaryTitle}>Transcript</div>
              <div
                style={{
                  ...styles.textSection,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  padding: '4px 0'
                }}
              >
                {reportData.transcriptSegments.split('\n').map((line, idx) => {
                  const m = line.match(/^\[(.*?)\]\s*(.*)/) || [];
                  const timestamp = m[1] || '';
                  const text = m[2] || '';
                  const emo = emotionPerSegment[idx] || 'None';
                  const color = emotionColors[emo] || '#888';

                  return (
                    <div
                      key={idx}
                      data-tooltip-id="emo-tooltip"
                      data-tooltip-content={emo}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        padding: '8px',
                        borderRadius: '6px',
                        background: isExporting ? '#fff' : '#fafafa',
                        boxShadow: isExporting ? 'none' : '0 1px 3px rgba(0,0,0,0.05)',
                        borderLeft: `6px solid ${color}`,
                        cursor: isExporting ? 'default' : 'pointer',
                        gap: isExporting ? '0' : '8px'
                      }}
                    >
                      {/* Timestamp pill */}
                      <div
                        style={{
                          flexShrink: 0,
                          fontSize: '0.75em',
                          fontWeight: '600',
                          color: isExporting ? '#000' : '#5D2E8C',
                          background: isExporting ? 'none' : 'rgba(107,76,175,0.1)',
                          borderRadius: '4px',
                          padding: '2px 6px',
                          lineHeight: 1.2,
                          marginRight: '12px',
                          textAlign: 'center'
                        }}
                      >
                        {timestamp}
                      </div>

                      {/* Transcript text */}
                      <div
                        style={{
                          color: '#333',
                          fontSize: '0.9em',
                          lineHeight: '1.4'
                        }}
                      >
                        {text}
                      </div>
                    </div>
                  )
                })}

                {/* Single tooltip instance — must be outside the map */}
                <RTTooltip id="emo-tooltip" place="top" float />
              </div>

              <div style={styles.legendContainer}>
                {Object.entries(emotionColors).map(([emotion, color]) => (
                  <div key={emotion} style={styles.legendItem}>
                    <div style={{ ...styles.legendColor, backgroundColor: color }} />
                    <div style={styles.legendLabel}>{emotion}</div>
                  </div>
                ))}
              </div>
            </div>
            <div style={styles.breakdownSection}>
              <div style={styles.summaryTitle}>Speech Improvement Assistance</div>
              <div style={styles.suggestionText}>
                {reportData.speechImprovements}
              </div>
            </div>
                        {/* NEW: AUDIO ANALYSIS SECTION */}
            <div style={styles.audioAnalysisSection}>
              <div style={styles.audioAnalysisTitle}>🎵 Advanced Audio Analysis</div>
              <div style={styles.audioAnalysisDescription}>
                Get detailed vocal dynamics analysis, pitch variation insights, and enhanced audio coaching based on your video's speech patterns.
              </div>
              <button
                style={{
                  ...styles.audioAnalysisButton,
                  ...(isProcessingAudio ? styles.audioAnalysisButtonDisabled : {})
                }}
                onClick={handleAudioAnalysis}
                disabled={isProcessingAudio}
                onMouseOver={(e) => {
                  if (!isProcessingAudio) {
                    e.currentTarget.style.backgroundColor = '#7248A0';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }
                }}
                onMouseOut={(e) => {
                  if (!isProcessingAudio) {
                    e.currentTarget.style.backgroundColor = '#5D2E8C';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }
                }}
              >
                <FaMicrophone />
                {isProcessingAudio ? 'Analysing Audio...' : 'Generate Audio Analysis'}
              </button>
            </div>
            <div style={styles.divider}></div>
            {/* SCORE BREAKDOWN SECTION */}
            <div style={styles.breakdownSection}>
              <div style={styles.breakdownTitle}>Score Breakdown</div>

              <div style={styles.breakdownContent}>
                {/* Face Emotion Analysis */}
                <div style={styles.placeholderBox}>
                  <div style={styles.graphTitle}>Face Emotion Analysis</div>
                  {/* Face Emotion Analysis Section */}
                  <section style={{ margin: '5px 0' }}>
                    <div
                      className="scroll-chart"
                      style={{
                        width: '100%',
                        overflowX: 'scroll',
                        overflowY: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          minWidth: '600px',
                          height: '60px',
                          borderRadius: '8px',
                          overflow: 'hidden',
                        }}
                      >
                        {(() => {
                          let accumulatedWidth = 0;
                          return segments.map((seg, i) => {
                            let widthPct;
                            if (i < segments.length - 1) {
                              widthPct = ((seg.end - seg.start) / totalDur) * 100;
                              accumulatedWidth += widthPct;
                            } else {
                              widthPct = 100 - accumulatedWidth;
                            }
                            const emo = emoSeg[i] || 'None';
                            const timeStart = `${seg.start.toFixed(2)}s`;
                            const timeEnd = `${seg.end.toFixed(2)}s`;
                            const tipId = `tip-${i}`;

                            return (
                              <div
                                key={i}
                                data-tooltip-id={tipId}
                                data-tooltip-content={`${emo}: ${timeStart} – ${timeEnd}`}
                                style={{
                                  width: `${widthPct}%`,
                                  height: '100%',
                                  backgroundColor: emotionColors[emo] || '#000',
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  padding: '4px',
                                  boxSizing: 'border-box',
                                  borderRight:
                                    i < segments.length - 1 ? '1px solid rgba(255,255,255,0.5)' : 'none',
                                  borderTopLeftRadius: i === 0 ? '8px' : '0',
                                  borderBottomLeftRadius: i === 0 ? '8px' : '0',
                                  borderTopRightRadius: i === segments.length - 1 ? '8px' : '0',
                                  borderBottomRightRadius: i === segments.length - 1 ? '8px' : '0'
                                }}
                              >
                                <span
                                  style={{
                                    fontSize: '0.8em',
                                    fontWeight: '600',
                                    lineHeight: 1.2,
                                    textAlign: 'center',
                                    color: '#fff',
                                    marginBottom: '2px'
                                  }}
                                >
                                  {emo}
                                </span>
                                <RTTooltip id={tipId} place="top" />
                              </div>
                            );
                          });
                        })()}
                      </div>
                    </div>
                  </section>

                  {/* Video Start / End labels */}
                  <div style={styles.barLabels}>
                    <span style={styles.barLabel}>Video Start</span>
                    <span style={styles.barLabel}>Video End</span>
                  </div>

                  <div style={styles.legendContainer}>
                    {Object.entries(emotionColors).map(([emotion, color]) => (
                      <div key={emotion} style={styles.legendItem}>
                        <div style={{ ...styles.legendColor, backgroundColor: color }} />
                        <div style={styles.legendLabel}>{emotion}</div>
                      </div>
                    ))}
                  </div>

                  <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                  <div style={styles.suggestionText}>{reportData.emotionText}</div>
                </div>

                {/* Graph 2 */}
                <div style={styles.placeholderBox}>
                  <div style={styles.graphTitle}>Movement Analysis</div>

                  {/* Horizontally scrollable chart container */}
                  <div
                    className="scroll-chart"
                    style={{
                      width: '100%',
                      overflowX: 'scroll',
                      overflowY: 'hidden',
                    }}
                  >
                    <div style={{ minWidth: '600px', height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={movementData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="second"
                            tickFormatter={(val, index) => {
                              if (index === 4) return 'Video Start';
                              if (index === movementData.length - 8) return 'Video End';
                              return '';
                            }}
                          />
                          <YAxis
                            domain={[0, 10]}
                            ticks={[1, 3, 5, 7, 9]}
                            tickFormatter={(val) => {
                              if (val === 1) return 'Left';
                              if (val === 3) return 'Middle Left';
                              if (val === 5) return 'Center';
                              if (val === 7) return 'Middle Right';
                              if (val === 9) return 'Right';
                              return '';
                            }}
                          />
                          <Tooltip
                            labelFormatter={(label) => `Time: ${label}`}
                            formatter={(value, name, props) =>
                              [`${value.toFixed(2)} (${props.payload.label})`, 'Position']
                            }
                          />
                          <Line type="monotone" dataKey="position" stroke="#5D2E8C" dot />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                  <div style={styles.suggestionText}>{reportData.movementText}</div>
                </div>

                <div className="breakdown-chart-pair" style={{ ...styles.breakdownContent, flexDirection: 'row', justifyContent: 'space-between' }}>
                  {/* Shoulder Posture Chart */}
                  <div style={{ ...styles.placeholderBox, flex: 1 }}>
                    <div style={styles.graphTitle}>Shoulder Posture Analysis</div>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={shoulderData}
                          dataKey="value"
                          nameKey="name"
                          outerRadius={60}
                          label={({ value }) => `${value}%`}
                        >
                          {shoulderData.map((entry, index) => (
                            <Cell key={`cell-shoulder-${index}`} fill={pieColors[index % pieColors.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value}%`} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                    <div style={styles.suggestionText}>{reportData.shoulderText}</div>
                  </div>

                  {/* Hand Gesture Chart */}
                  <div style={{ ...styles.placeholderBox, flex: 1 }}>
                    <div style={styles.graphTitle}>Hand Gestures Analysis</div>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={handsData}
                          dataKey="value"
                          nameKey="name"
                          outerRadius={60}
                          label={({ value }) => `${value}%`}
                        >
                          {handsData.map((entry, index) => (
                            <Cell key={`cell-hands-${index}`} fill={pieColors[index % pieColors.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value}%`} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                    <div style={styles.suggestionText}>{reportData.gestureText}</div>
                  </div>

                </div>
                {/* Graph 3 */}
                <div style={styles.placeholderBox}>
                  <div style={styles.graphTitle}>Gaze Analysis</div>
                  <div style={{
                    position: 'relative',
                    width: '220px',
                    height: '220px',
                    margin: '20px auto'
                  }}>
                    {/* SVG background */}
                    <div style={{
                      position: 'absolute',
                      width: '70%',
                      height: '110%',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      zIndex: 0
                    }}>
                      <Mansvg style={{
                        width: '100%',
                        height: '100%',
                        fill: '#5D2E8C',
                        opacity: 1,
                        stroke: '#5D2E8C',
                      }} />
                    </div>

                    {/* Heatmap grid */}
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gridTemplateRows: 'repeat(3, 1fr)',
                      width: '100%',
                      height: '100%',
                      zIndex: 1
                    }}>
                      {['up left', 'up', 'up right',
                        'left', 'center', 'right',
                        'down left', 'down', 'down right'].map((pos) => (
                          <div
                            key={pos}
                            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
                            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                            style={{
                              backgroundColor: getHeatColor(gazePercentages[pos] || 0),
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '14px',
                              fontWeight: '600',
                              color: '#fff',
                              textShadow: '0 1px 3px rgba(0,0,0,0.8)',
                              position: 'relative',
                              cursor: 'pointer',
                              transition: 'transform 0.2s ease, background-color 0.3s ease',
                              borderRadius: '12px',
                              padding: '6px',
                              margin: '2px'
                            }}
                          >
                            <div style={{
                              fontSize: '16px',
                              fontWeight: '700'
                            }}>
                              {gazePercentages[pos]}%
                            </div>
                            <div style={{
                              fontSize: '12px',
                              marginTop: '2px'
                            }}>
                              {pos}
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div style={styles.suggestionsTitle}>Suggestions for Improvement</div>
                  <div style={styles.suggestionText}>{reportData.gazeText}</div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
        <button className="download-button" onClick={handlePrint}>
          <FaDownload />
          Download Report
        </button>
      </div>
      {/* --- MODAL --- */}
      {modalOpen && (
        <div style={styles.backdrop} onClick={() => setModalOpen(false)}>
          <div style={styles.modal} onClick={e => e.stopPropagation()}>
            <h2 style={{ margin: '0 0 1rem 0' }}>Share & Download</h2>

            {/* QR code preview */}
            {imgDataUrl && (
              <div style={styles.qrWrap}>
                <QRCode value={imgDataUrl} size={160} />
              </div>
            )}

            <p style={{ fontSize: '.9rem', marginBottom: '1.5rem' }}>
              Scan on another device or click below to save the file.
            </p>

            <button className="dl-btn" style={styles.dlBtn} onClick={downloadNow}>
              Download now
            </button>
            <button className="close-btn" style={styles.closeBtn} onClick={() => setModalOpen(false)}>
              Close
            </button>
          </div>
        </div>
      )}
      <style>{`

.breakdown-chart-pair {
  flex-direction: row;
  display: flex;
  gap: 20px;
  justify-content: space-between;
  flex-wrap: wrap;
}

@media (max-width: 768px) {
  .breakdown-chart-pair {
    flex-direction: column !important;
  }
}

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
        /* card grid stacks naturally; no extra rule needed */
        /* download button full width on mobile */
        @media (max-width: 480px) {
          button { width: 100% !important; justify-content: center !important; }
        }

.dl-btn,
.close-btn {
  transition: background 0.25s ease, transform 0.2s ease, box-shadow 0.2s ease;
}

/* Download button hover */
.dl-btn:hover {
  background: #4b2175;           /* slightly darker brand colour */
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,.15);
}

/* Close button hover */
.close-btn:hover {
  background: #f5f5f5;
}
.scroll-chart {
  scrollbar-color: #5D2E8C transparent; /* Firefox */
  scrollbar-width: auto;
    border-radius: 8px;
  overflow-x: scroll !important;
}



`}</style>
    </div>
  );
}