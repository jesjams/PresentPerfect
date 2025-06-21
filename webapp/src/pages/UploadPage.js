import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { ReactComponent as UploadIcon } from '../assets/upload.svg';
import { ReactComponent as CheckBox } from '../assets/checkbox.svg';
import { ReactComponent as UploadFile } from '../assets/uploadfile.svg';
import { ReactComponent as DropVideo } from '../assets/videodrop.svg';
import { io } from 'socket.io-client';
import { useAuth } from '../context/AuthContext';

export default function UploadPage() {
  const navigate = useNavigate();

  const [toast, setToast] = useState(null);
  const [toastVisible, setToastVisible] = useState(false);

  const socketRef = useRef(null);

  const [processing, setProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingMsg, setProcessingMsg] = useState('');

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [reportData, setReportData] = useState(null);

  const { user, login } = useAuth();
  const [pendingNavigate, setPendingNavigate] = useState(false);
  const [emailInput, setEmailInput] = useState('');
  const [passwordInput, setPasswordInput] = useState('');
  const [loginError, setLoginError] = useState('');

  useEffect(() => {
    socketRef.current = io('http://localhost:4000');

    socketRef.current.on('processing-update', data => {
      setProcessingProgress(data.progress);
      setProcessingMsg(data.message);
    });

    socketRef.current.on('processing-complete', data => {
      setProcessingProgress(100);
      console.log(data)
      setProcessingMsg(data.message);
      setReportData(data.data);
      setProcessing(false);
      if (user) {
        navigate('/report', { state: { reportData: data.data } });
      } else {
        setPendingNavigate(true);
      }
    });

    return () => socketRef.current.disconnect();
  }, [navigate, user, reportData]);


  useEffect(() => {
    if (user && pendingNavigate && reportData) {
      navigate('/report', { state: { reportData } });
    }
  }, [user, pendingNavigate, reportData, navigate]);


  const showToast = message => {
    setToast(message);
    setToastVisible(true);
    setTimeout(() => {
      setToastVisible(false);
      setTimeout(() => setToast(null), 400);
    }, 3000);
  };


  const handleInlineLogin = e => {
    e.preventDefault();
    if (login(emailInput, passwordInput)) {
      setLoginError('');
    } else {
      setLoginError('Invalid credentials.');
    }
  };


  const onDrop = useCallback(async (acceptedFiles, fileRejections) => {
    if (fileRejections.length) {
      showToast('Please upload a valid .mp4 video file.');
      return;
    }
    if (!acceptedFiles.length) return;

    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append('video', file);

    try {
      setUploading(true);
      await axios.post('http://localhost:4000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: evt => {
          const pct = Math.round((evt.loaded / evt.total) * 100);
          setUploadProgress(pct);
        }
      });
      setUploading(false);
      setProcessing(true);
      setProcessingProgress(0);
      setProcessingMsg('Starting analysis…');
    } catch (err) {
      console.error(err);
      showToast('Upload failed');
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { 'video/mp4': [] },
    multiple: false,
    noClick: true,
    noKeyboard: true
  });


  const styles = {
    /* page */
    page: {
      padding: '1rem',
      backgroundColor: '#f9f9ff',
      position: 'relative',
      zIndex: 1
    },
    heading: {
      textAlign: 'center',
      fontSize: '2.5rem',
      marginBottom: '.5rem'
    },

    /* upload */
    uploadBox: {
      boxSizing: 'border-box',
      width: '90%',
      minHeight: '300px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#5D2E8C',
      borderRadius: '1.5rem',
      margin: '0 auto 1rem',
      position: 'relative',
      boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
      outline: '2px dashed rgba(255,255,255,0.0)',
      outlineOffset: '-2px',
      zIndex: 2,
      transition: 'outline-color 0.2s ease'
    },
    innerBorder: {
      position: 'absolute',
      top: '12px',
      bottom: '12px',
      left: '12px',
      right: '12px',
      border: '2px dashed rgba(255,255,255,0.6)',
      borderRadius: '1.25rem',
      pointerEvents: 'none',
      transition: 'border-color 0.2s ease, border-style 0.2s ease'
    },
    dragOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0,0,0,0.6)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: '1.5rem',
      zIndex: 3,
      pointerEvents: 'none',
      color: 'white',
      textAlign: 'center'
    },

    /* progress overlay */
    progressOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0,0,0,0.7)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      borderRadius: '1.5rem',
      zIndex: 4,
      pointerEvents: 'none',
      color: 'white',
      textAlign: 'center'
    },
    progressMessage: {
      fontSize: '1.2rem',
      marginBottom: '1rem'
    },
    progressWrapper: {
      width: '80%',
      height: '1.2rem',
      backgroundColor: 'rgba(255,255,255,0.3)',
      borderRadius: '0.6rem',
      overflow: 'hidden'
    },
    progressBar: {
      height: '100%',
      borderRadius: '999px',
      backgroundImage: 'repeating-linear-gradient(135deg, #B288C0 0 20px, #ffffff 10px 30px)',
      backgroundSize: '40px 40px',
      backgroundPosition: '0 0',
      animation: 'candy 1s linear infinite',
      transition: 'width 1s ease-out',
    },
    progressPercent: {
      marginTop: '0.5rem',
      fontVariantNumeric: 'tabular-nums'
    },

    /* upload content */
    uploadContent: {
      position: 'relative',
      zIndex: 2,
      textAlign: 'center',
      justifyItems: 'center',
      color: 'white'
    },
    icon: { fill: '#f9f9ff' },
    uploadButton: {
      marginTop: '1rem',
      color: 'black',
      fontSize: '1.1rem',
      padding: '0.8rem 2rem',
      borderRadius: '1rem',
      border: 'none',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      transition: 'all 0.5s ease-in-out'
    },

    /* info section */
    section: {
      display: 'flex',
      gap: '2rem',
      margin: '0 auto 3rem',
      marginLeft: '5%',
      position: 'relative',
      zIndex: 2
    },
    instructions: { flex: 1, fontSize: '1rem', lineHeight: '1.6' },
    checklist: { flex: 1, fontSize: '1rem' },
    checkItem: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      marginBottom: '1rem'
    },
    checkIcon: { color: '#5e2b97', fontSize: '1.2rem' },

    /* triangle bg */
    triangleBg: {
      position: 'fixed',
      top: '40%',
      right: 0,
      width: '100%',
      height: '60%',
      opacity: '10%',
      backgroundColor: '#B288C0',
      zIndex: 0,
      clipPath: 'polygon(70% 15%, 0% 100%, 85% 100%)',
      pointerEvents: 'none'
    },

    /* toast */
    toast: {
      position: 'fixed',
      bottom: '2rem',
      left: '50%',
      transform: 'translateX(-50%)',
      backgroundColor: '#5D2E8C',
      color: 'white',
      padding: '1rem 2rem',
      borderRadius: '1rem',
      boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
      zIndex: 999,
      fontSize: '1rem',
      animation: `${toastVisible ? 'fadeSlideIn' : 'fadeSlideOut'} 0.4s forwards`,
      pointerEvents: 'none'
    },
    loginOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '95.5%',
      height: '79%',
      backgroundColor: 'rgba(255, 255, 255, 0.29)',
      borderRadius: '1.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
      zIndex: 5
    },
    loginCard: {
      backgroundColor: '#f9f9ff',
      padding: '2rem',
      borderRadius: '1.25rem',
      boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
      width: '100%',
      height: '90%',
      maxWidth: '350px',
      textAlign: 'left',
    },
    loginTitle: {
      fontSize: '1.5rem',
      color: '#5D2E8C',
      marginBottom: '1rem'
    },
    loginInput: {
      width: '100%',
      padding: '0.7rem',
      marginBottom: '1rem',
      borderRadius: '0.75rem',
      border: '1px solid #ccc',
      fontSize: '1rem',
      outline: 'none',
      boxSizing: 'border-box'
    },
    loginButton: {
      width: '100%',
      padding: '0.8rem',
      fontSize: '1rem',
      borderRadius: '1rem',
      border: 'none',
      cursor: 'pointer',
      backgroundColor: '#5D2E8C',
      color: 'white',
      transition: 'background 0.3s ease-in-out'
    },
    loginButtonHover: {
      backgroundColor: '#B288C0'
    },
    loginError: {
      color: 'red',
      marginBottom: '1rem',
      fontWeight: '500'
    }
  };

  const activeProgress = uploading ? uploadProgress : processingProgress;
  const activeMessage = uploading ? 'Uploading video…' : processingMsg;

  return (
    <div style={styles.page}>
      <h1 style={styles.heading}>Analyse Yourself</h1>

      <div {...getRootProps()} style={styles.uploadBox}>
        <input {...getInputProps()} />
        <div style={styles.innerBorder} />

        {isDragActive && (
          <div style={styles.dragOverlay}>
            <DropVideo style={{ width: '5rem', height: '5rem', stroke: '#fff' }} />
            <p style={{ marginTop: '1rem', fontSize: '1.2rem' }}>
              Drop it like it's HOT
            </p>
          </div>
        )}

        {processing && (
          <div style={styles.progressOverlay}>
            <p style={styles.progressMessage}>{activeMessage}</p>
            <div style={styles.progressWrapper}>
              <div style={{ ...styles.progressBar, width: `${activeProgress}%` }} />
            </div>
            <p style={styles.progressPercent}>{activeProgress}%</p>
          </div>
        )}

        {!isDragActive && !uploading && !processing && (
          <div style={styles.uploadContent}>
            <div style={styles.icon}>
              <UploadIcon style={{ width: '8rem', height: '8rem', stroke: '#f9f9ff' }} />
            </div>
            <div>Drag and drop your video to upload</div>
            <button type="button" onClick={open} style={styles.uploadButton}>
              <UploadFile style={{ width: '1.5rem', height: '1.5rem', stroke: '#000' }} />
              Choose Video
            </button>
          </div>
        )}

        {/* INLINE LOGIN PROMPT */}
        {pendingNavigate && !user && (
          <div style={styles.loginOverlay}>
            <div style={styles.loginCard}>
              <h2 style={styles.loginTitle}>Please log in to see your analysis</h2>
              {loginError && <div style={styles.loginError}>{loginError}</div>}
              <form onSubmit={handleInlineLogin}>
                <input
                  type="email"
                  placeholder="Email"
                  value={emailInput}
                  onChange={e => setEmailInput(e.target.value)}
                  style={styles.loginInput}
                  required
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={passwordInput}
                  onChange={e => setPasswordInput(e.target.value)}
                  style={styles.loginInput}
                  required
                />
                <button
                  type="submit"
                  style={styles.loginButton}
                  onMouseOver={e => (e.currentTarget.style.backgroundColor = styles.loginButtonHover.backgroundColor)}
                  onMouseOut={e => (e.currentTarget.style.backgroundColor = styles.loginButton.backgroundColor)}
                >
                  Log In
                </button>
              </form>
            </div>
          </div>
        )}
      </div>

      {/* info section */}
      <div style={styles.section}>
        <div style={styles.instructions}>
          <h2>Instructions</h2>
          <p>
            Nervous about speaking in front of a crowd? Don’t worry — we’ve got your
            back. Simply upload a short video of yourself speaking. Our AI will
            analyze your posture, gestures, and movement patterns to provide
            insightful feedback.
            <br />
            &nbsp; &nbsp; &nbsp;Make sure your entire upper body is clearly visible
            <br />
            &nbsp; &nbsp; &nbsp;Record in a well-lit environment with minimal
            background clutter
            <br />
            Once submitted, your results will appear with suggestions to help
            improve your delivery.
          </p>
        </div>

        <div style={styles.checklist}>
          <h2>Why use PresentPerfect?</h2>
          {['Real-time posture analysis to correct body language issues like slouching',
            'Confidence feedback by detecting emotion, movement and stance',
            'Practice smarter — get actionable insights instantly for real presentations'].map((text, i) => (
              <div key={i} style={styles.checkItem}>
                <span style={styles.checkIcon}>
                  <CheckBox style={{ width: '1.5rem', height: '1.5rem', fill: '#5D2E8C' }} />
                </span>
                <span>{text}</span>
              </div>
            ))}
        </div>
      </div>

      {/* background triangle */}
      <div style={styles.triangleBg} />

      {/* toast */}
      {toast && <div style={styles.toast}>{toast}</div>}
    </div>
  );
}
