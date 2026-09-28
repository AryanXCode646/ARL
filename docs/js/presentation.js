/**
 * ARL - Adaptive Reinforcement Learning
 * College Demonstration & Viva Presentation Mode
 */

(function () {
  'use strict';

  let currentSlide = 0;
  const totalSlides = 10;
  let modal, progressEl, prevBtn, nextBtn, exitBtn, fullscreenBtn;

  function init() {
    modal = document.getElementById('presentation-modal');
    progressEl = document.getElementById('pres-progress-text');
    prevBtn = document.getElementById('pres-btn-prev');
    nextBtn = document.getElementById('pres-btn-next');
    exitBtn = document.getElementById('pres-btn-exit');
    fullscreenBtn = document.getElementById('pres-btn-fullscreen');

    const triggerBtn = document.getElementById('btn-open-presentation');
    if (triggerBtn) {
      triggerBtn.addEventListener('click', openPresentation);
    }

    const heroPresBtn = document.getElementById('hero-btn-presentation');
    if (heroPresBtn) {
      heroPresBtn.addEventListener('click', openPresentation);
    }

    if (prevBtn) prevBtn.addEventListener('click', prevSlide);
    if (nextBtn) nextBtn.addEventListener('click', nextSlide);
    if (exitBtn) exitBtn.addEventListener('click', closePresentation);

    if (fullscreenBtn) {
      fullscreenBtn.addEventListener('click', toggleFullscreen);
    }

    window.addEventListener('keydown', (e) => {
      if (!modal || !modal.classList.contains('active')) return;

      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        nextSlide();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevSlide();
      } else if (e.key === 'Escape') {
        closePresentation();
      }
    });
  }

  function openPresentation() {
    if (!modal) return;
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    currentSlide = 0;
    showSlide(currentSlide);
  }

  function closePresentation() {
    if (!modal) return;
    modal.classList.remove('active');
    document.body.style.overflow = '';
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  }

  function showSlide(index) {
    const slides = document.querySelectorAll('.pres-slide');
    slides.forEach((s, idx) => {
      if (idx === index) {
        s.classList.add('active');
      } else {
        s.classList.remove('active');
      }
    });

    if (progressEl) {
      progressEl.textContent = `Slide ${index + 1} of ${totalSlides}`;
    }

    if (prevBtn) prevBtn.disabled = (index === 0);
    if (nextBtn) {
      nextBtn.textContent = (index === totalSlides - 1) ? 'Finish' : 'Next →';
    }
  }

  function nextSlide() {
    if (currentSlide < totalSlides - 1) {
      currentSlide++;
      showSlide(currentSlide);
    } else {
      closePresentation();
    }
  }

  function prevSlide() {
    if (currentSlide > 0) {
      currentSlide--;
      showSlide(currentSlide);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
