// Brain Tumor Detection Application JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const fileInput = document.getElementById('fileInput');
    const uploadedImage = document.getElementById('uploadedImage');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    const resultDiv = document.getElementById('result');
    const uploadBox = document.getElementById('uploadBox');
    const tumorCountElement = document.getElementById('tumorCount');
    const noTumorCountElement = document.getElementById('noTumorCount');
    
    // Fetch dataset statistics
    fetchDatasetInfo();
    
    // Set up drag and drop functionality
    uploadBox.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });
    
    uploadBox.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
    });
    
    uploadBox.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    // Handle file selection via click
    fileInput.addEventListener('change', function(event) {
        if (fileInput.files.length) {
            handleFileSelect(fileInput.files[0]);
        }
    });
    
    // Handle file selection (both drag and click)
    function handleFileSelect(file) {
        if (file) {
            // Validate file is an image
            if (!file.type.match('image.*')) {
                showResult('<span class="error-message"><i class="fas fa-exclamation-circle"></i> Please select a valid image file!</span>', true);
                uploadedImageContainer.style.display = 'none';
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
                
                // Show the uploaded image and enable analysis button
                uploadedImageContainer.style.display = 'block';
                startAnalysisBtn.style.display = 'flex';
                resultDiv.style.display = 'none'; // Hide previous results
            };
            
            reader.onerror = function() {
                showResult('<span class="error-message"><i class="fas fa-exclamation-circle"></i> Error reading the file!</span>', true);
            };
            
            reader.readAsDataURL(file);
        }
    }
    
    // Handle analysis button click
    startAnalysisBtn.addEventListener('click', function() {
        // Reset result and show loading
        showResult('<div class="loader"></div> Analyzing your MRI scan...', true);
        startAnalysisBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        startAnalysisBtn.disabled = true;
        
        const formData = new FormData();
        const file = fileInput.files[0];
        
        if (file) {
            formData.append("image", file);
            
            // Send to backend for analysis
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Network response was not ok');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Process successful response
                const isTumor = data.result.includes('Brain Tumour');
                const iconClass = isTumor ? 'fa-exclamation-triangle' : 'fa-check-circle';
                const resultClass = isTumor ? 'error-message' : 'success-message';
                
                let resultHTML = `
                    <div class="${resultClass}">
                        <i class="fas ${iconClass}" style="font-size: 2rem; margin-bottom: 15px;"></i>
                        <h3>${data.result}</h3>
                    </div>`;
                
                // Add confidence information if available
                if (data.confidence) {
                    resultHTML += `<div style="margin-top: 15px;">Confidence: <strong>${data.confidence}</strong></div>`;
                }
                
                showResult(resultHTML, true);
                startAnalysisBtn.innerHTML = '<i class="fas fa-microscope"></i> Start Analysis';
                startAnalysisBtn.disabled = false;
            })
            .catch(error => {
                showResult(`<span class="error-message"><i class="fas fa-exclamation-circle"></i> Error: ${error.message}</span>`, true);
                startAnalysisBtn.innerHTML = '<i class="fas fa-microscope"></i> Start Analysis';
                startAnalysisBtn.disabled = false;
            });
        } else {
            showResult('<span class="error-message"><i class="fas fa-exclamation-circle"></i> No file uploaded!</span>', true);
            startAnalysisBtn.innerHTML = '<i class="fas fa-microscope"></i> Start Analysis';
            startAnalysisBtn.disabled = false;
        }
    });
    
    // Function to show result
    function showResult(html, display) {
        resultDiv.innerHTML = html;
        resultDiv.style.display = display ? 'block' : 'none';
    }
    
    // Function to fetch dataset statistics
    function fetchDatasetInfo() {
        fetch('/dataset-info')
            .then(response => response.json())
            .then(data => {
                tumorCountElement.textContent = `Tumor Cases: ${data.tumor_images}`;
                noTumorCountElement.textContent = `No Tumor Cases: ${data.no_tumor_images}`;
            })
            .catch(error => {
                tumorCountElement.textContent = 'Stats unavailable';
                noTumorCountElement.textContent = 'Stats unavailable';
                console.error('Error fetching dataset info:', error);
            });
    }
    
    // Initial UI setup
    uploadedImageContainer.style.display = 'none';
    resultDiv.style.display = 'none';
});