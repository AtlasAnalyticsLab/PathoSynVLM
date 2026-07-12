(() => {
  "use strict";

  const header = document.querySelector("[data-site-header]");
  const navToggle = document.querySelector("[data-nav-toggle]");
  const navigation = document.querySelector("[data-navigation]");

  const updateHeader = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 8);
  };

  const closeNavigation = () => {
    if (!navToggle || !navigation) return;
    navToggle.setAttribute("aria-expanded", "false");
    navigation.classList.remove("is-open");
  };

  navToggle?.addEventListener("click", () => {
    const isOpen = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!isOpen));
    navigation?.classList.toggle("is-open", !isOpen);
  });

  navigation?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeNavigation);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNavigation();
  });

  window.addEventListener("scroll", updateHeader, { passive: true });
  updateHeader();

  const copyButton = document.querySelector("[data-copy-citation]");
  const copyLabel = document.querySelector("[data-copy-label]");
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
    if (!citation || !copyLabel || !copyStatus) return;

    try {
      const text = citation.textContent.trim();
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        legacyCopy(text);
      }
      copyLabel.textContent = "Copied";
      copyStatus.textContent = "BibTeX copied to the clipboard.";
      window.setTimeout(() => {
        copyLabel.textContent = "Copy BibTeX";
        copyStatus.textContent = "";
      }, 2200);
    } catch {
      copyStatus.textContent = "Copy failed. Select the citation text manually.";
    }
  });

  const year = document.querySelector("[data-current-year]");
  if (year) year.textContent = String(new Date().getFullYear());
})();
