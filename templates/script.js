document.addEventListener("DOMContentLoaded", function () {

    // IMAGE BUTTON
    document.getElementById("imageBtn").addEventListener("click", function () {

        let file = document.getElementById("imageInput").files[0];

        if (!file) {
            alert("Please select an image");
            return;
        }

        let formData = new FormData();
        formData.append("image", file);

        fetch("/predict_image", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById("imageResult").innerText = data.error;
            } else {
                document.getElementById("imageResult").innerHTML =
                    "Result: " + data.result + " (" + data.confidence + "%)<br>" +
                    "Metadata: " + data.metadata;
            }
        })
        .catch(error => {
            document.getElementById("imageResult").innerText = "Error processing image.";
            console.error(error);
        });

    });


    // VIDEO BUTTON
    document.getElementById("videoBtn").addEventListener("click", function () {

        let file = document.getElementById("videoInput").files[0];

        if (!file) {
            alert("Please select a video");
            return;
        }

        let formData = new FormData();
        formData.append("video", file);

        fetch("/predict_video", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById("videoResult").innerText = data.error;
            } else {
                document.getElementById("videoResult").innerHTML =
                    "Result: " + data.result + " (" + data.confidence + "%)";
            }
        })
        .catch(error => {
            document.getElementById("videoResult").innerText = "Error processing video.";
            console.error(error);
        });

    });

});
