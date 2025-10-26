document.addEventListener('DOMContentLoaded', function () {
    // Initialize AOS (Animate on Scroll)
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 800, // Animation duration
            once: true      // Animation happens only once
        });
    } else {
        console.log('AOS library not loaded.');
    }

    // Initialize Bootstrap Tooltips (for the info icons on the predict page)
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // We are using standard HTML form submissions for prediction,
    // so no complex fetch logic is needed here anymore.
    console.log("DoseRight AI frontend script loaded. Animations and tooltips initialized.");
});