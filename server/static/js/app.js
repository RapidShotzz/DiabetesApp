document.addEventListener("DOMContentLoaded", function() {
    const ranges = document.querySelectorAll('input[type="range"]');

    ranges.forEach(range => {
        const id = range.id;
        const valueLabel = document.getElementById(`${id}-value`);
        valueLabel.textContent = range.value;

        range.addEventListener('input', function() {
            // Ensure to format the value correctly for floats
            valueLabel.textContent = parseFloat(this.value).toFixed(1);
        });
    });
});

function predict() {
    const data = {
        Pregnancies: parseFloat(document.getElementById('pregnancies').value),
        Glucose: parseFloat(document.getElementById('glucose').value),
        BloodPressure: parseFloat(document.getElementById('bloodpressure').value),
        SkinThickness: parseFloat(document.getElementById('skinthickness').value),
        Insulin: parseFloat(document.getElementById('insulin').value),
        BMI: parseFloat(document.getElementById('BMI').value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById('diabetespedigreefunction').value),
        Age: parseFloat(document.getElementById('age').value),
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        const predictionText = result.prediction === 1 ? 'Diabetic' : 'Non-Diabetic';
        document.getElementById('prediction-result').textContent = `Prediction: ${predictionText}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('prediction-result').textContent = 'An error occurred during prediction.';
    });
}