(() => {
  "use strict";

  const navToggle = document.querySelector(".nav-toggle");
  const siteNav = document.querySelector("#site-navigation");

  const closeNavigation = () => {
    navToggle?.setAttribute("aria-expanded", "false");
    siteNav?.classList.remove("is-open");
  };

  navToggle?.addEventListener("click", () => {
    const willOpen = navToggle.getAttribute("aria-expanded") !== "true";
    navToggle.setAttribute("aria-expanded", String(willOpen));
    siteNav?.classList.toggle("is-open", willOpen);
  });

  siteNav?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeNavigation);
  });

  document.addEventListener("click", (event) => {
    if (!siteNav?.classList.contains("is-open")) return;
    if (siteNav.contains(event.target) || navToggle?.contains(event.target)) return;
    closeNavigation();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !siteNav?.classList.contains("is-open")) return;
    closeNavigation();
    navToggle?.focus();
  });

  window.matchMedia("(min-width: 821px)").addEventListener("change", (event) => {
    if (event.matches) closeNavigation();
  });

  const tabs = Array.from(document.querySelectorAll("[role='tab']"));

  const activateTab = (tab, moveFocus = true) => {
    tabs.forEach((candidate) => {
      const selected = candidate === tab;
      candidate.setAttribute("aria-selected", String(selected));
      candidate.tabIndex = selected ? 0 : -1;

      const panel = document.getElementById(candidate.dataset.tabTarget);
      panel?.classList.toggle("is-active", selected);
    });

    if (moveFocus) tab.focus();
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab, false));
    tab.addEventListener("keydown", (event) => {
      let nextIndex;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;
      if (nextIndex === undefined) return;

      event.preventDefault();
      activateTab(tabs[nextIndex]);
    });
  });

  const lightbox = document.querySelector("#lightbox");
  const lightboxImage = lightbox?.querySelector("img");
  const lightboxClose = lightbox?.querySelector(".lightbox-close");

  document.querySelectorAll("[data-lightbox-src]").forEach((trigger) => {
    trigger.addEventListener("click", () => {
      if (!lightbox || !lightboxImage) return;
      lightboxImage.src = trigger.dataset.lightboxSrc;
      lightboxImage.alt = trigger.dataset.lightboxAlt || "Expanded paper figure";
      lightbox.showModal();
    });
  });

  lightboxClose?.addEventListener("click", () => lightbox?.close());
  lightbox?.addEventListener("click", (event) => {
    if (event.target === lightbox) lightbox.close();
  });

  const copyButton = document.querySelector("[data-copy-citation]");
  const copyStatus = document.querySelector("[data-copy-status]");
  const citation = document.querySelector("#bibtex");

  const legacyCopy = (text) => {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.setAttribute("readonly", "");
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.append(textArea);
    textArea.select();
    const succeeded = document.execCommand("copy");
    textArea.remove();
    if (!succeeded) throw new Error("Copy command failed");
  };

  copyButton?.addEventListener("click", async () => {
    if (!citation || !copyStatus) return;

    try {
      const text = citation.textContent.trim();
      if (navigator.clipboard && window.isSecureContext) {
        try {
          await navigator.clipboard.writeText(text);
        } catch {
          legacyCopy(text);
        }
      } else {
        legacyCopy(text);
      }

      copyButton.textContent = "Copied!";
      copyButton.classList.add("is-copied");
      copyStatus.textContent = "BibTeX copied to the clipboard.";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
        copyButton.classList.remove("is-copied");
        copyStatus.textContent = "";
      }, 2000);
    } catch {
      copyStatus.textContent = "Copy failed. Select the citation text manually.";
    }
  });
})();
