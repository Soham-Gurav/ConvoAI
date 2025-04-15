const socket = io();

// DOM elements
const messagesDiv = document.getElementById("messages");
const userInput = document.getElementById("user_input");
const sendButton = document.getElementById("send_button");
const callButton = document.getElementById("call_button");
const setKbButton = document.getElementById("set_kb_button");
const kbSelect = document.getElementById("kb_select");


let recognition;
let isListening = false;
let userSpeechTimeout;
let mediaRecorder;
let audioChunks = [];
let mediaStream;  // To store the media stream

// Log WebSocket connection status
socket.on("connect", () => {
    console.log("✅ Connected to WebSocket server.");
});

socket.on("disconnect", () => {
    console.log("❌ Disconnected from WebSocket server.");
});

if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript;
        userInput.value = transcript;
        sendMessage(transcript, "call");  // Use "call" mode for timed input
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
    };

    recognition.onend = () => {
        if (isListening) {
            // Restart recognition after a short delay
            setTimeout(() => recognition.start(), 100);
        }
    };
} else {
    alert("Your browser does not support speech recognition.");
}

// Start recording user audio
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
            mediaStream = stream;
            
            // Check WAV support
            if (!MediaRecorder.isTypeSupported('audio/wav')) {
                console.warn('WAV recording not natively supported');
            }
            
            // Configure recorder
            const options = {
                mimeType: 'audio/wav',
                audioBitsPerSecond: 128000
            };
            
            try {
                mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                console.warn('Using browser default recording format:', e);
                mediaRecorder = new MediaRecorder(stream);
            }
            
            mediaRecorder.start();
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, {
                    type: mediaRecorder.mimeType || 'audio/wav'
                });
                
                const audioFile = new File([audioBlob], "user_audio.wav", {
                    type: 'audio/wav'
                });
                
                const formData = new FormData();
                formData.append("audio", audioFile);
                fetch("/save_user_audio", { method: "POST", body: formData })
                    .then((response) => {
                        if (response.ok) console.log("WAV audio saved successfully");
                        else console.error("Failed to save audio");
                    });
                
                audioChunks = [];
            };
        })
        .catch((error) => console.error("Mic access error:", error));
}

// Stop recording user audio and release the media stream
function stopRecording() {
    if (mediaRecorder) {
        mediaRecorder.stop();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());  // Stop all tracks in the media stream
        mediaStream = null;  // Clear the media stream
    }
}

// Start the 5-second input window
function startInputWindow() {
    isListening = true;
    recognition.start();
    startRecording();  // Start recording user audio
    console.log("🎤 Listening for user input...");

    // Set a 5-second timeout for user input
    userSpeechTimeout = setTimeout(() => {
        isListening = false;
        recognition.stop();
        stopRecording();  // Stop recording user audio
        console.log("⏹️ Input window closed.");
        sendMessage("", "call");  // Send an empty message to trigger AI response
    }, 5000);  // 5-second window
}

// Stop the 5-second input window
function stopInputWindow() {
    isListening = false;
    recognition.stop();
    stopRecording();  // Stop recording user audio
    clearTimeout(userSpeechTimeout);  // Clear the 5-second timeout
    console.log("⏹️ Input window stopped.");
}

// Send message to the backend
function sendMessage(message, mode) {
    const systemPrompt = document.getElementById("system_prompt").value;
    if (message && message.trim()) {  // Only send non-empty messages
        socket.emit("user_message", { message, mode, systemPrompt });
        messagesDiv.innerHTML += `<p><strong>You:</strong> ${message}</p>`;
        userInput.value = "";
    } else {
        console.log("⏹️ No user input detected. Skipping AI response.");
    }
}

// Handle AI response
socket.on("ai_response", (data) => {
    messagesDiv.innerHTML += `<p><strong>AI:</strong> ${data.message}</p>`;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (data.mode === "call" && data.audio_file) {
        const audio = new Audio(data.audio_file);
        audio.play().catch((error) => console.error("Error playing audio:", error));

        // Wait for the AI's audio to finish playing
        audio.onended = () => {
            console.log("🎧 AI audio finished playing.");
            // Stop the microphone and speech recognition
            stopInputWindow();
            // Restart the 5-second input window after AI finishes speaking
            startInputWindow();
        };
    }
});

// Handle call ending
socket.on("call_ended", (data) => {
    console.log("📞 Call ended:", data.message);
    callButton.textContent = "Start Call";
    callButton.style.backgroundColor = "#28a745"; // Green color for "Start Call"
    stopInputWindow(); // Ensure mic stops
});

// Start Call button
callButton.addEventListener("click", () => {
    if (!isListening) {
        callButton.textContent = "End Call";
        callButton.style.backgroundColor = "#dc3545";  // Red color for "End Call"
        startInputWindow();  // Start the 5-second input window
    } else {
        socket.emit("call_ended"); // Notify the backend
        callButton.textContent = "Start Call";
        callButton.style.backgroundColor = "#28a745";  // Green color for "Start Call"
        stopInputWindow();  // Stop the 5-second input window
    }
});


// Send Message button
sendButton.addEventListener("click", () => {
    const message = userInput.value.trim();
    if (message) {
        sendMessage(message, "text");  // Use "text" mode for manual input
    } else {
        console.log("❌ No message to send.");
    }
});

setKbButton.addEventListener("click", () => {
    const kbFolder = kbSelect.value;
    fetch("/set_knowledge_base", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kb_folder: kbFolder })
    })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error("Error setting knowledge base:", error));
});

// Analyze Call button
/*
document.getElementById("analyze_button").addEventListener("click", () => {
    fetch("/analyze_call", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }
        
        // Display analysis results in a formatted way
        const analysisResults = `
            <h3>Call Analysis Results</h3>
            <h4>Transcript Summary</h4>
            <p>Total turns: ${data.transcript.length}</p>
            <p>User speaking time: ${data.speaker_times.user.toFixed(1)}s</p>
            <p>AI speaking time: ${data.speaker_times.ai.toFixed(1)}s</p>
            
            <h4>Sentiment Analysis</h4>
            <p>Positive turns: ${data.sentiment.positive}</p>
            <p>Neutral turns: ${data.sentiment.neutral}</p>
            <p>Negative turns: ${data.sentiment.negative}</p>
            <p>Overall sentiment: ${data.sentiment.overall_sentiment.toFixed(2)}</p>
            
            <h4>Word Count</h4>
            <p>User words: ${data.word_counts.user}</p>
            <p>AI words: ${data.word_counts.ai}</p>
            
            <h4>Top Topics</h4>
            <ul>
                ${Object.entries(data.topics).map(([topic, count]) => 
                    `<li>${topic}: ${count} mentions</li>`
                ).join('')}
            </ul>
            
            <h4>Full Transcript</h4>
            <div class="transcript">
                ${data.transcript.map(turn => 
                    `<p><strong>${turn.speaker}:</strong> ${turn.text} (${turn.time.toFixed(1)}s)</p>`
                ).join('')}
            </div>
        `;
        
        // Create a modal to display results
        const modal = document.createElement('div');
        modal.className = 'analysis-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close">&times;</span>
                ${analysisResults}
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close button functionality
        modal.querySelector('.close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    })
    .catch(error => {
        console.error("Error analyzing call:", error);
        alert("Error analyzing call. See console for details.");
    });
});
*/
// Analyze Call button
document.getElementById("analyze_button").addEventListener("click", () => {
    fetch("/analyze_call", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }
        
        // Fetch the summary after analysis
        fetch("/generate_call_summary", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        })
        .then(response => response.json())
        .then(summaryData => {
            if (summaryData.error) {
                alert(`Error: ${summaryData.error}`);
                return;
            }
            
            // Display analysis results in a formatted way
            const analysisResults = `
                <h3>Call Analysis Results</h3>
                <h4>Transcript Summary</h4>
                <p>Total turns: ${data.transcript.length}</p>
                <p>User speaking time: ${data.speaker_times.user.toFixed(1)}s</p>
                <p>AI speaking time: ${data.speaker_times.ai.toFixed(1)}s</p>
                
                <h4>Sentiment Analysis</h4>
                <p>Positive turns: ${data.sentiment.positive}</p>
                <p>Neutral turns: ${data.sentiment.neutral}</p>
                <p>Negative turns: ${data.sentiment.negative}</p>
                <p>Overall sentiment: ${data.sentiment.overall_sentiment.toFixed(2)}</p>
                
                <h4>Word Count</h4>
                <p>User words: ${data.word_counts.user}</p>
                <p>AI words: ${data.word_counts.ai}</p>
                
                <h4>Top Topics</h4>
                <ul>
                    ${Object.entries(data.topics).map(([topic, count]) => 
                        `<li>${topic}: ${count} mentions</li>`
                    ).join('')}
                </ul>
                
                <h4>Summary</h4>
                <p>${summaryData.summary}</p>
                
                <h4>Full Transcript</h4>
                <div class="transcript">
                    ${data.transcript.map(turn => 
                        `<p><strong>${turn.speaker}:</strong> ${turn.text} (${turn.time.toFixed(1)}s)</p>`
                    ).join('')}
                </div>
            `;
            
            // Create a modal to display results
            const modal = document.createElement('div');
            modal.className = 'analysis-modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <span class="close">&times;</span>
                    ${analysisResults}
                </div>
            `;
            
            document.body.appendChild(modal);
            
            // Close button functionality
            modal.querySelector('.close').addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        })
        .catch(error => {
            console.error("Error generating summary:", error);
            alert("Error generating summary. See console for details.");
        });
    })
    .catch(error => {
        console.error("Error analyzing call:", error);
        alert("Error analyzing call. See console for details.");
    });
});
