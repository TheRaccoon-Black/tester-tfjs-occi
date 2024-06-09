const CLASS_NAMES = ['Apel', 'Brokoli', 'Jeruk', 'Kangkung', 'Mangga', 'Pisang', 'Strawberry', 'Terong', 'Toge', 'Wortel'];
let model;

async function loadModel() {
    console.log('Loading model...');
    try {
        model = await tf.loadLayersModel('DenseNet201/model.json'); // Update the path according to your directory structure
        console.log('Model berhasil dimuat');
    } catch (error) {
        console.error('Gagal memuat model', error);
    }
}

async function classifyImage() {
    const imageUpload = document.getElementById('image-upload');
    if (imageUpload.files.length === 0) {
        alert('Silakan unggah gambar terlebih dahulu!');
        console.log('No image uploaded');
        return;
    }

    if (!model) {
        alert('Model belum dimuat. Silakan tunggu dan coba lagi.');
        console.log('Model not loaded');
        return;
    }

    const file = imageUpload.files[0];
    const img = new Image();
    const reader = new FileReader();

    console.log('Reading the uploaded image...');
    reader.onload = function(event) {
        img.src = event.target.result;
        img.onload = async function() {
            console.log('Image loaded, processing the image...');
            const tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([416, 416]) // Update to the size your model expects
                .toFloat()
                .expandDims(0); // Add batch dimension

            console.log('Image tensor created:', tensor);
            try {
                const prediction = model.predict(tensor);
                console.log('Prediction tensor:', prediction);
                const probabilities = prediction.dataSync(); // Get the probabilities for each class
                console.log('Probabilities:', probabilities);
                const predictedIndex = prediction.argMax(-1).dataSync()[0];
                console.log('Predicted index:', predictedIndex);
                const predictedClass = CLASS_NAMES[predictedIndex];
                console.log('Predicted class:', predictedClass);
                document.getElementById('result').innerText = `Prediksi: ${predictedClass}`;

                // Display probabilities for each class
                const probabilitiesElement = document.getElementById('probabilities');
                probabilitiesElement.innerHTML = '<h3>Tingkat Akurasi Tiap Kelas:</h3>';
                probabilitiesElement.innerHTML += '<ul class="list-group">';
                CLASS_NAMES.forEach((className, index) => {
                    probabilitiesElement.innerHTML += `<li class="list-group-item">${className}: ${(probabilities[index] * 100).toFixed(2)}%</li>`;
                });
                probabilitiesElement.innerHTML += '</ul>';
            } catch (error) {
                console.error('Gagal melakukan prediksi', error);
            }
        };
    };
    reader.readAsDataURL(file);
}

document.getElementById('classify-button').addEventListener('click', classifyImage);

loadModel();
