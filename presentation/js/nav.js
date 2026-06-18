document.addEventListener("DOMContentLoaded", function () {
  if (typeof Reveal === "undefined") {
    return;
  }

  var nav = document.getElementById("section-nav");
  if (!nav) {
    return;
  }

  var topLevelSections = document.querySelectorAll(".reveal .slides > section");
  var sectionCount = 0;

  topLevelSections.forEach(function (topSection, hIndex) {
    var level1Slide = topSection.matches(".slide.level1, section.level1")
      ? topSection
      : topSection.querySelector("section.slide.level1, section.level1");

    if (!level1Slide) {
      return;
    }

    var titleElement = level1Slide.querySelector("h1, h2");
    var title = titleElement ? titleElement.textContent.trim() : "";
    if (!title) {
      return;
    }

    var sectionKey = level1Slide.dataset.section || level1Slide.id;
    if (!sectionKey) {
      sectionKey = "section-" + (sectionCount + 1);
      level1Slide.dataset.section = sectionKey;
    }

    var navItem = document.createElement("button");
    navItem.type = "button";
    navItem.className = "section-nav-item";
    navItem.dataset.section = sectionKey;
    navItem.textContent = title;
    navItem.addEventListener("click", function () {
      Reveal.slide(hIndex, 0);
    });
    nav.appendChild(navItem);
    sectionCount += 1;
  });

  if (!sectionCount) {
    nav.style.display = "none";
    return;
  }

  function getCurrentSectionKey(currentSlide) {
    if (!currentSlide) {
      return null;
    }

    var level1Slide = currentSlide.matches(".slide.level1, section.level1")
      ? currentSlide
      : currentSlide
          .closest("section.stack")
          ?.querySelector("section.slide.level1, section.level1");

    if (!level1Slide) {
      return null;
    }

    return level1Slide.dataset.section || level1Slide.id || null;
  }

  function updateSection() {
    var currentSectionKey = getCurrentSectionKey(Reveal.getCurrentSlide());

    nav.querySelectorAll(".section-nav-item").forEach(function (item) {
      item.classList.toggle("active", item.dataset.section === currentSectionKey);
    });
  }

  Reveal.on("ready", updateSection);
  Reveal.on("slidechanged", updateSection);
  updateSection();
});