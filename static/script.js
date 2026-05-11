document.addEventListener("DOMContentLoaded", function () {

    // ================= IMAGE PREDICTION =================
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

            console.log("Image Response:", data);

            if (data.error) {
                document.getElementById("imageResult").innerHTML =
                    "Error: " + data.error;
            } else {
                document.getElementById("imageResult").innerHTML =
                    "Result: " + data.result + " (" + data.confidence + "%)<br>" +
                    "Metadata: " + (data.metadata || "Not Available");
            }
        })
        .catch(error => {
            console.error("Image Fetch Error:", error);
            document.getElementById("imageResult").innerHTML =
                "Error processing image.";
        });

    });


    // ================= VIDEO PREDICTION =================
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

            console.log("Video Response:", data);

            if (data.error) {
                document.getElementById("videoResult").innerHTML =
                    "Error: " + data.error;
            } else {
                document.getElementById("videoResult").innerHTML =
                    "Result: " + data.result + " (" + data.confidence + "%)";
            }
        })
        .catch(error => {
            console.error("Video Fetch Error:", error);
            document.getElementById("videoResult").innerHTML =
                "Error processing video.";
        });

    });

});