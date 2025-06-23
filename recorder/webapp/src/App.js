import React, { useState, useRef, useEffect } from 'react';

/*
🏗️ DISTRIBUTED ARCHITECTURE SETUP:

📱 MOBILE PHONES → 🖥️ L3-UI → 🖥️ L1-BACKEND → 🖥️ L2-UI

L1 (Main Backend Server):
- Runs main app's backend (analysis processing)  
- Runs Python file_receiver.py on port 3001
- Receives uploaded recordings from phones

L2 (Main App Frontend):
- Runs main app's React UI 
- Connects to L1's backend for analysis results

L3 (Recording Frontend Server): 👈 THIS APP RUNS HERE
- Runs this recording app UI on port 3000
- Mobile phones connect via: http://L3-IP:3000
- Uploads files to L1-backend via: http://L1-IP:3001

SETUP INSTRUCTIONS:
1. L1: Run `python file_receiver.py` + main app backend
2. L2: Run main app's React UI
3. L3: Run `npm start` (this recording app)
4. Phones: Browse to http://L3-IP:3000

⚙️ CONFIGURATION:
Create config.json in project root:
{
  "mode": "local",
  "local": {
    "uploadUrl": "http://localhost:3001/upload"
  },
  "distributed": {
    "uploadUrl": "http://192.168.1.100:3001/upload"
  }
}
*/


const UPLOAD_URL = 'https://9fea-202-65-71-17.ngrok-free.app/upload';

const RecordingApp = () => {
  // State management
  const [currentStep, setCurrentStep] = useState('topics'); // 'topics', 'countdown', 'recording', 'complete'
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [recordingMode, setRecordingMode] = useState('video'); // 'video' or 'audio'
  const [countdown, setCountdown] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('');
  const [showQR, setShowQR] = useState(false);

  // Refs for media recording
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const videoPreviewRef = useRef(null);
  const chunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const countdownTimerRef = useRef(null);
  const recordingStartTime = useRef(null);

  // Get current URL for QR code - use local IP, not derived from UPLOAD_URL
  const getCurrentUrl = () => {
    return 'https://192.168.68.121:3000'; 
  };

  // Recording topics with bullet points
  const topics = [
    {
      id: 'presentation',
      title: 'Presentation Skills',
      bullets: [
        'Introduce yourself and your topic',
        'Explain why this topic matters',
        'Share your main point clearly',
        'Conclude with a call to action'
      ]
    },
    {
      id: 'elevator',
      title: 'Elevator Pitch',
      bullets: [
        'State who you are and what you do',
        'Mention your unique value proposition',
        'Share a brief success story',
        'End with what you\'re looking for'
      ]
    },
    {
      id: 'product',
      title: 'Product Demo',
      bullets: [
        'Introduce the product and its purpose',
        'Demonstrate key features',
        'Highlight benefits for users',
        'Mention pricing or next steps'
      ]
    },
    {
      id: 'interview',
      title: 'Job Interview',
      bullets: [
        'Give a brief professional introduction',
        'Explain your relevant experience',
        'Share why you\'re interested in this role',
        'Ask a thoughtful question'
      ]
    },
    {
      id: 'free',
      title: 'Free Speech',
      bullets: [
        'Choose any topic you\'re passionate about',
        'Speak naturally and authentically',
        'Focus on clear communication',
        'Let your personality shine through'
      ]
    }
  ];

  // Styles matching the main app
  const styles = {
    container: {
      backgroundColor: '#f9f9f9',
      minHeight: '100vh',
      padding: '20px',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    },
    header: {
      textAlign: 'center',
      marginBottom: '2rem'
    },
    title: {
      fontSize: '2.5rem',
      color: '#5D2E8C',
      marginBottom: '0.5rem',
      fontWeight: '700'
    },
    subtitle: {
      fontSize: '1.2rem',
      color: '#666',
      marginBottom: '2rem'
    },
    mainCard: {
      backgroundColor: '#5D2E8C',
      borderRadius: '20px',
      padding: '10px',
      maxWidth: '900px',
      margin: '0 auto',
      boxShadow: '0 8px 24px rgba(93, 46, 140, 0.3)',
      width: '95%' // Better mobile fit
    },
    innerCard: {
      backgroundColor: 'white',
      borderRadius: '15px',
      padding: '20px', // Reduced padding for mobile
      border: '2px dashed #5D2E8C'
    },
    topicGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', // Smaller minimum for mobile
      gap: '15px', // Reduced gap
      marginBottom: '2rem'
    },
    topicCard: {
      backgroundColor: '#f8f9fa',
      border: '2px solid #e9ecef',
      borderRadius: '15px',
      padding: '20px',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      textAlign: 'center'
    },
    topicCardSelected: {
      backgroundColor: '#5D2E8C',
      color: 'white',
      borderColor: '#5D2E8C',
      transform: 'translateY(-4px)',
      boxShadow: '0 8px 20px rgba(93, 46, 140, 0.3)'
    },
    topicTitle: {
      fontSize: '1.3rem',
      fontWeight: '600',
      marginBottom: '0.5rem'
    },
    modeToggle: {
      display: 'flex',
      justifyContent: 'center',
      gap: '1rem',
      marginBottom: '2rem'
    },
    modeButton: {
      padding: '12px 24px',
      borderRadius: '25px',
      border: '2px solid #5D2E8C',
      backgroundColor: 'white',
      color: '#5D2E8C',
      cursor: 'pointer',
      fontSize: '1rem',
      fontWeight: '600',
      transition: 'all 0.3s ease'
    },
    modeButtonActive: {
      backgroundColor: '#5D2E8C',
      color: 'white',
      transform: 'translateY(-2px)',
      boxShadow: '0 4px 12px rgba(93, 46, 140, 0.3)'
    },
    startButton: {
      backgroundColor: '#5D2E8C',
      color: 'white',
      border: 'none',
      padding: '15px 40px',
      borderRadius: '25px',
      fontSize: '1.2rem',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      boxShadow: '0 4px 12px rgba(93, 46, 140, 0.3)',
      margin: '0 auto',
      display: 'block'
    },
    countdownContainer: {
      textAlign: 'center',
      padding: '2rem'
    },
    countdownNumber: {
      fontSize: '6rem',
      fontWeight: 'bold',
      color: '#5D2E8C',
      lineHeight: 1,
      marginBottom: '1rem'
    },
    bulletsContainer: {
      backgroundColor: '#f8f9fa',
      borderRadius: '15px',
      padding: '20px',
      marginBottom: '2rem',
      textAlign: 'left'
    },
    bulletItem: {
      fontSize: '1.1rem',
      marginBottom: '0.8rem',
      padding: '8px 0',
      borderBottom: '1px solid #e9ecef',
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
    },
    recordingContainer: {
      textAlign: 'center',
      padding: '2rem'
    },
    videoPreview: {
      width: '100%',
      maxWidth: '300px',
      height: '200px',
      borderRadius: '15px',
      border: '3px solid #5D2E8C',
      marginBottom: '1rem',
      backgroundColor: '#000'
    },
    recordingTimer: {
      fontSize: '3rem',
      fontWeight: 'bold',
      color: '#5D2E8C',
      marginBottom: '2rem',
      fontFamily: 'monospace'
    },
    recordButton: {
      width: '120px',
      height: '120px',
      borderRadius: '50%',
      border: 'none',
      cursor: 'pointer',
      fontSize: '1.2rem',
      fontWeight: '600',
      transition: 'all 0.3s ease',
      marginBottom: '1rem'
    },
    recordButtonRecord: {
      backgroundColor: '#dc3545',
      color: 'white',
      boxShadow: '0 0 20px rgba(220, 53, 69, 0.5)'
    },
    recordButtonStop: {
      backgroundColor: '#28a745',
      color: 'white',
      boxShadow: '0 0 20px rgba(40, 167, 69, 0.5)'
    },
    statusText: {
      fontSize: '1.1rem',
      color: '#5D2E8C',
      fontWeight: '500',
      marginBottom: '1rem'
    },
    successMessage: {
      backgroundColor: '#d4edda',
      color: '#155724',
      padding: '20px',
      borderRadius: '10px',
      fontSize: '1.1rem',
      textAlign: 'center',
      marginBottom: '2rem'
    },
    resetButton: {
      backgroundColor: '#6c757d',
      color: 'white',
      border: 'none',
      padding: '12px 30px',
      borderRadius: '25px',
      fontSize: '1rem',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.3s ease'
    },
    qrButton: {
      position: 'fixed',
      top: '10px',
      right: '10px',
      backgroundColor: '#5D2E8C',
      color: 'white',
      border: 'none',
      padding: '8px 15px', // Smaller on mobile
      borderRadius: '20px',
      fontSize: '0.9rem',
      fontWeight: '600',
      cursor: 'pointer',
      boxShadow: '0 4px 12px rgba(93, 46, 140, 0.3)',
      zIndex: 1000
    },
    qrOverlay: {
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1001
    },
    qrModal: {
      backgroundColor: 'white',
      padding: '40px',
      borderRadius: '20px',
      textAlign: 'center',
      maxWidth: '400px',
      boxShadow: '0 12px 30px rgba(0, 0, 0, 0.3)'
    },
    qrCode: {
      margin: '20px 0',
      padding: '20px',
      backgroundColor: '#f8f9fa',
      borderRadius: '10px'
    }
  };

  // Check permissions and start countdown
  const startCountdown = async () => {
    if (!selectedTopic) {
      alert('Please select a topic first!');
      return;
    }
    
    // 🎥 CHECK PERMISSIONS FIRST - before countdown starts
    try {
      setStatus('Checking camera and microphone permissions...');
      
      const constraints = recordingMode === 'video' 
        ? { video: true, audio: true }
        : { audio: true };
      
      // Test permissions by requesting access
      const testStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Stop the test stream immediately
      testStream.getTracks().forEach(track => track.stop());
      
      // ✅ Permissions granted! Start countdown
      setStatus('');
      setCurrentStep('countdown');
      setCountdown(5);
      
      countdownTimerRef.current = setInterval(() => {
        setCountdown(prev => {
          if (prev <= 1) {
            clearInterval(countdownTimerRef.current);
            startRecording();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      
    } catch (error) {
      // ❌ Permission denied or device not available
      console.error('Permission error:', error);
      let errorMessage = 'Camera/microphone access required. ';
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Please allow access and try again.';
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera/microphone found.';
      } else {
        errorMessage += 'Please check your device settings.';
      }
      
      alert(errorMessage);
      setStatus('');
    }
  };

  // Start recording (permissions already checked)
  const startRecording = async () => {
    try {
      const constraints = recordingMode === 'video' 
        ? { video: true, audio: true }
        : { audio: true };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      // Show video preview if in video mode
      if (recordingMode === 'video' && videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        videoPreviewRef.current.play().catch(error => {
          console.log('Video preview play failed (normal on mobile):', error);
        });
      }
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: recordingMode === 'video' ? 'video/mp4' : 'audio/mp3'
        });
        uploadRecording(blob);
      };
      
      // Start recording and timer precisely
      mediaRecorder.start();
      setIsRecording(true);
      setCurrentStep('recording');
      setRecordingTime(0);
      recordingStartTime.current = Date.now();
      
      // Use more accurate timer based on actual elapsed time
      recordingTimerRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime.current) / 1000);
        setRecordingTime(elapsed);
      }, 100); // Update every 100ms for smoother display
      
    } catch (error) {
      // This should rarely happen since we pre-checked permissions
      console.error('Error starting recording:', error);
      alert('Recording failed to start. Please try again.');
      setCurrentStep('topics'); // Reset to topic selection
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      // Clear video preview
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = null;
      }
      
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    }
  };

  // Upload recording
  const uploadRecording = async (blob) => {
    const formData = new FormData();
    const now = new Date();
    const timestamp = now.getHours().toString().padStart(2, '0') +
                     now.getMinutes().toString().padStart(2, '0') +
                     now.getSeconds().toString().padStart(2, '0');
    const extension = recordingMode === 'video' ? 'mp4' : 'mp3';
    const filename = `recording_${timestamp}_${recordingMode}.${extension}`;
    
    formData.append('recording', blob, filename);
    
    // ✅ Jump directly to complete - no progress screen
    setCurrentStep('complete');
    
    // 💾 FALLBACK: Download file locally first
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    console.log('💾 File downloaded locally as backup:', filename);
    
    try {
      // 🔥 UPLOAD TO BACKEND
      console.log(`🚀 Attempting upload to: ${UPLOAD_URL}`);
      
      const response = await fetch(UPLOAD_URL, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        // 📝 Console logging for success tracking
        console.log('✅ RECORDING UPLOADED SUCCESSFULLY');
        console.log('📁 Filename:', filename);
        console.log('🎯 Sent to backend for analysis');
        console.log('⏰ Upload completed at:', new Date().toLocaleTimeString());
      } else {
        console.error('❌ Upload failed with status:', response.status);
        console.error('❌ Response:', await response.text());
        throw new Error(`Upload failed with status ${response.status}`);
      }
    } catch (error) {
      // 🚨 Console logging for errors
      console.error('❌ UPLOAD FAILED:', error.message);
      console.log('📁 Filename:', filename);
      console.log('💾 File was downloaded locally as backup');
      console.log('⚠️ Check if backend is running file_receiver.py on port 3001');
    }
  };

  // Reset to start
  const reset = () => {
    setCurrentStep('topics');
    setSelectedTopic(null);
    setRecordingMode('video');
    setCountdown(0);
    setRecordingTime(0);
    setIsRecording(false);
    setStatus('');
    
    // Clear any running timers
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    
    // Clean up video preview
    if (videoPreviewRef.current) {
      videoPreviewRef.current.srcObject = null;
    }
    
    // Stop any active streams
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  };

  // Format time helper
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = null;
      }
    };
  }, []);

  return (
    <div style={styles.container}>
      {/* QR Code Button */}
      <button
        style={styles.qrButton}
        onClick={() => setShowQR(true)}
        onMouseOver={(e) => e.target.style.backgroundColor = '#7248A0'}
        onMouseOut={(e) => e.target.style.backgroundColor = '#5D2E8C'}
      >
        📱 Show QR Code
      </button>

      {/* QR Code Modal */}
      {showQR && (
        <div style={styles.qrOverlay} onClick={() => setShowQR(false)}>
          <div style={styles.qrModal} onClick={(e) => e.stopPropagation()}>
            <h2 style={{color: '#5D2E8C', marginBottom: '1rem'}}>
              📱 Scan to Access on Phone
            </h2>
            <div style={styles.qrCode}>
              <img
                src={`https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(getCurrentUrl())}`}
                alt="QR Code"
                style={{maxWidth: '200px', height: '200px'}}
              />
            </div>
            <p style={{color: '#666', fontSize: '0.9rem', marginBottom: '1rem'}}>
              URL: {getCurrentUrl()}
            </p>
            <p style={{color: '#666', fontSize: '0.9rem', marginBottom: '1.5rem'}}>
              Make sure your phone is on the same WiFi network!
            </p>
            <button
              style={{
                ...styles.resetButton,
                backgroundColor: '#5D2E8C'
              }}
              onClick={() => setShowQR(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div style={styles.header}>
        <h1 style={styles.title}>🎤 Presentation Recorder</h1>
        <p style={styles.subtitle}>Practice your presentation skills with AI-powered analysis</p>
      </div>

      <div style={styles.mainCard}>
        <div style={styles.innerCard}>
          
          {/* Topic Selection */}
          {currentStep === 'topics' && (
            <>
              <h2 style={{textAlign: 'center', color: '#5D2E8C', marginBottom: '2rem'}}>
                Choose a Speaking Topic
              </h2>
              
              <div style={styles.topicGrid}>
                {topics.map(topic => (
                  <div
                    key={topic.id}
                    style={{
                      ...styles.topicCard,
                      ...(selectedTopic?.id === topic.id ? styles.topicCardSelected : {})
                    }}
                    onClick={() => setSelectedTopic(topic)}
                  >
                    <div style={styles.topicTitle}>{topic.title}</div>
                    <div style={{fontSize: '0.9rem', opacity: 0.8}}>
                      Click to see speaking points
                    </div>
                  </div>
                ))}
              </div>

              {selectedTopic && (
                <div style={styles.bulletsContainer}>
                  <h3 style={{color: '#5D2E8C', marginBottom: '1rem'}}>
                    Speaking Points for "{selectedTopic.title}":
                  </h3>
                  {selectedTopic.bullets.map((bullet, index) => (
                    <div key={index} style={styles.bulletItem}>
                      <span style={{color: '#5D2E8C', fontWeight: 'bold'}}>•</span>
                      {bullet}
                    </div>
                  ))}
                </div>
              )}

              <div style={styles.modeToggle}>
                <button
                  style={{
                    ...styles.modeButton,
                    ...(recordingMode === 'video' ? styles.modeButtonActive : {})
                  }}
                  onClick={() => setRecordingMode('video')}
                >
                  🎥 Video + Audio
                </button>
                <button
                  style={{
                    ...styles.modeButton,
                    ...(recordingMode === 'audio' ? styles.modeButtonActive : {})
                  }}
                  onClick={() => setRecordingMode('audio')}
                >
                  🎵 Audio Only
                </button>
              </div>

              <button
                style={{
                  ...styles.startButton,
                  ...(status ? {
                    backgroundColor: '#ccc',
                    cursor: 'not-allowed'
                  } : {})
                }}
                onClick={startCountdown}
                disabled={status !== ''}
                onMouseOver={(e) => !status && (e.target.style.backgroundColor = '#7248A0')}
                onMouseOut={(e) => !status && (e.target.style.backgroundColor = '#5D2E8C')}
              >
                {status ? status : 'Start Recording Session'}
              </button>
            </>
          )}

          {/* Countdown */}
          {currentStep === 'countdown' && (
            <div style={styles.countdownContainer}>
              <h2 style={{color: '#5D2E8C', marginBottom: '1rem'}}>Get Ready!</h2>
              
              <div style={styles.countdownNumber}>{countdown}</div>
              <p style={{fontSize: '1.2rem', color: '#666'}}>
                Recording will start automatically...
              </p>
              {selectedTopic && (
                <div style={styles.bulletsContainer}>
                  <h4 style={{color: '#5D2E8C', marginBottom: '1rem'}}>Remember to cover:</h4>
                  {selectedTopic.bullets.map((bullet, index) => (
                    <div key={index} style={styles.bulletItem}>
                      <span style={{color: '#28a745', fontWeight: 'bold'}}>✓</span>
                      {bullet}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Recording */}
          {currentStep === 'recording' && (
            <div style={styles.recordingContainer}>
              <h2 style={{color: '#5D2E8C', marginBottom: '1rem'}}>🔴 Recording in Progress</h2>
              
              {/* Video Preview - only show for video mode */}
              {recordingMode === 'video' && (
                <div style={{marginBottom: '1rem'}}>
                  <video
                    ref={videoPreviewRef}
                    style={styles.videoPreview}
                    muted
                    playsInline
                    autoPlay
                    controls={false}
                  />
                  <p style={{fontSize: '0.8rem', color: '#666', marginTop: '0.5rem'}}>
                    Preview may not work on all mobile browsers - recording still works fine!
                  </p>
                </div>
              )}
              
              <div style={styles.recordingTimer}>{formatTime(recordingTime)}</div>
              
              <button
                style={{
                  ...styles.recordButton,
                  ...styles.recordButtonStop
                }}
                onClick={stopRecording}
                onMouseOver={(e) => e.target.style.transform = 'scale(1.1)'}
                onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
              >
                STOP<br/>RECORDING
              </button>
              
              <p style={{fontSize: '1rem', color: '#666', marginTop: '1rem'}}>
                Click the button above when finished speaking
              </p>

              {/* Speaking points during recording */}
              {selectedTopic && (
                <div style={{...styles.bulletsContainer, marginTop: '2rem'}}>
                  <h4 style={{color: '#5D2E8C', marginBottom: '1rem'}}>
                    {selectedTopic.id === 'free' ? 'General Speaking Tips:' : `Remember to cover (${selectedTopic.title}):`}
                  </h4>
                  {(selectedTopic.id === 'free' ? [
                    'Speak clearly and at a steady pace',
                    'Make eye contact with your audience',
                    'Use natural gestures to emphasize points',
                    'Conclude with a strong final thought'
                  ] : selectedTopic.bullets).map((bullet, index) => (
                    <div key={index} style={styles.bulletItem}>
                      <span style={{color: '#5D2E8C', fontWeight: 'bold'}}>•</span>
                      {bullet}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Complete - No more uploading screen */}
          {currentStep === 'complete' && (
            <div style={{textAlign: 'center'}}>
              <div style={styles.successMessage}>
                🎉 Recording Complete! Thank you!
              </div>
              <button
                style={styles.resetButton}
                onClick={reset}
                onMouseOver={(e) => e.target.style.backgroundColor = '#5a6268'}
                onMouseOut={(e) => e.target.style.backgroundColor = '#6c757d'}
              >
                Record Another Session
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default RecordingApp;