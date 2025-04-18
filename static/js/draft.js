window.addEventListener('load', () => {
    // Reset all slots & pool on page load
    document.querySelectorAll('.draft-slot').forEach(s => s.innerHTML = '');
    document.querySelectorAll('.hero-icon.disabled').forEach(i => i.classList.remove('disabled'));

    // Track current active slot
    let activeSlot = null;
    let currentPhase = 'ban';
    let currentTeam = 'blue';
    let banIndex = 0;
    let pickIndex = 0;

    function advanceSlot() {
        if (currentPhase === 'ban') {
            banIndex++;
            if (banIndex >= 3) {
                banIndex = 0;
                if (currentTeam === 'red') {
                    currentPhase = 'pick';
                    currentTeam = 'blue';
                } else {
                    currentTeam = 'red';
                }
            }
        } else {
            pickIndex++;
            if (pickIndex >= 5) {
                pickIndex = 0;
                currentTeam = currentTeam === 'blue' ? 'red' : 'blue';
            }
        }

        // Update UI
        document.getElementById('phase-text').textContent = `${currentPhase.charAt(0).toUpperCase() + currentPhase.slice(1)} Phase`;
        document.getElementById('turn-text').textContent = `${currentTeam.charAt(0).toUpperCase() + currentTeam.slice(1)} Team's Turn`;
        
        // Highlight active slot
        if (activeSlot) activeSlot.classList.remove('active');
        const selector = `.${currentTeam === 'blue' ? 'ally' : 'opp'}-panel .${currentPhase}-slot:nth-child(${currentPhase === 'ban' ? banIndex + 1 : pickIndex + 1})`;
        activeSlot = document.querySelector(selector);
        if (activeSlot) activeSlot.classList.add('active');
    }

    function slugify(text) {
        return text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    }

    function fillSlot(slot, hero) {
        const heroSlug = slugify(hero);
        slot.innerHTML = `
            <img src="/static/img/${heroSlug}.png" class="slot-img" 
                 onerror="this.src='/static/img/placeholder.png'">
            <button class="clear-btn">&times;</button>`;
        slot.classList.add('filled');
        document.querySelector(`.hero-icon[data-hero="${hero}"]`).classList.add('disabled');
        slot.querySelector('.clear-btn').onclick = () => clearSlot(slot, hero);
        advanceSlot();
    }

    function clearSlot(slot, hero) {
        slot.innerHTML = '';
        slot.classList.remove('filled');
        document.querySelector(`.hero-icon[data-hero="${hero}"]`).classList.remove('disabled');
    }

    // Setup drag and drop
    document.querySelectorAll('.hero-icon').forEach(icon => {
        icon.addEventListener('dragstart', e => {
            if (!icon.classList.contains('disabled')) {
                e.dataTransfer.setData('text/plain', icon.dataset.hero);
            }
        });

        icon.addEventListener('click', () => {
            if (!icon.classList.contains('disabled') && activeSlot) {
                fillSlot(activeSlot, icon.dataset.hero);
            }
        });
    });

    document.querySelectorAll('.draft-slot').forEach(slot => {
        slot.addEventListener('dragover', e => e.preventDefault());
        slot.addEventListener('drop', e => {
            e.preventDefault();
            if (slot === activeSlot) {
                const hero = e.dataTransfer.getData('text/plain');
                fillSlot(slot, hero);
            }
        });
    });

    // Setup search and filter
    const searchInput = document.getElementById('heroSearch');
    const roleFilter = document.getElementById('roleFilter');

    function filterHeroes() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedRole = roleFilter.value;

        document.querySelectorAll('.hero-icon').forEach(icon => {
            const heroName = icon.dataset.hero.toLowerCase();
            const heroRole = icon.dataset.role;
            const matchesSearch = heroName.includes(searchTerm);
            const matchesRole = selectedRole === 'All Roles' || heroRole === selectedRole;

            icon.style.display = matchesSearch && matchesRole ? '' : 'none';
        });
    }

    searchInput.addEventListener('input', filterHeroes);
    roleFilter.addEventListener('change', filterHeroes);

    // Initialize first active slot
    advanceSlot();
});

function getPhaseTimer(phase, pickIndex) {
    if (phase === 'ban') return 30;
    if (phase === 'pick') {
        return pickIndex === 0 ? 45 : 35;
    }
    return 30; // default
}

// Draft state management
class DraftState {
    constructor() {
        this.phase = 'ban';  // 'ban' or 'pick'
        this.turn = 1;       // 1-10
        this.bans = { blue: [], red: [] };
        this.picks = { blue: [], red: [] };
        this.timer = 30;     // seconds
        this.selectedHero = null;
        this.selectedSlot = null;
    }

    reset() {
        this.phase = 'ban';
        this.turn = 1;
        this.bans = { blue: [], red: [] };
        this.picks = { blue: [], red: [] };
        this.timer = 30;
        this.selectedHero = null;
        this.selectedSlot = null;
    }

    serialize() {
        return btoa(JSON.stringify({
            phase: this.phase,
            turn: this.turn,
            bans: this.bans,
            picks: this.picks
        }));
    }

    deserialize(data) {
        try {
            const state = JSON.parse(atob(data));
            Object.assign(this, state);
        } catch (e) {
            console.error('Failed to deserialize state:', e);
            throw new Error('Invalid draft state data');
        }
    }
}

// Storage management with fallback
class Storage {
    static set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.warn('localStorage not available, falling back to sessionStorage', e);
            try {
                sessionStorage.setItem(key, JSON.stringify(value));
            } catch (e) {
                console.error('Storage not available', e);
            }
        }
    }

    static get(key) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : null;
        } catch (e) {
            console.warn('localStorage not available, falling back to sessionStorage', e);
            try {
                const item = sessionStorage.getItem(key);
                return item ? JSON.parse(item) : null;
            } catch (e) {
                console.error('Storage not available', e);
                return null;
            }
        }
    }
}

// Toast notifications
class Toast {
    static container = document.querySelector('.toast-container');
    
    static show(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.setAttribute('role', 'alert');
        
        this.container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Draft simulator
class DraftSimulator {
    constructor() {
        this.state = new DraftState();
        this.heroCards = document.querySelectorAll('.hero-card');
        this.draftSlots = document.querySelectorAll('.draft-slot');
        this.searchInput = document.querySelector('#hero-search');
        this.roleFilter = document.querySelector('#role-filter');
        this.timerElement = document.querySelector('#timer');
        this.phaseElement = document.querySelector('#phase');
        
        this.setupEventListeners();
        this.loadState();
        this.startTimer();
    }

    init() {
        this.setupSearchAndFilter();
        this.setupDragAndDrop();
        this.setupKeyboardSupport();
        this.loadStateFromHash();
    }

    setupEventListeners() {
        // Hero card events
        this.heroCards.forEach(card => {
            card.addEventListener('click', (e) => this.handleHeroClick(e));
            card.addEventListener('keydown', (e) => this.handleHeroKeydown(e));
        });

        // Draft slot events
        this.draftSlots.forEach(slot => {
            slot.addEventListener('click', (e) => this.handleSlotClick(e));
            slot.addEventListener('keydown', (e) => this.handleSlotKeydown(e));
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleGlobalKeydown(e));
    }

    setupSearchAndFilter() {
        if (this.searchInput) {
            this.searchInput.addEventListener('input', () => this.filterHeroes());
        }
        
        if (this.roleFilter) {
            this.roleFilter.addEventListener('change', () => this.filterHeroes());
        }
    }

    filterHeroes() {
        const searchTerm = this.searchInput?.value.toLowerCase() || '';
        const selectedRole = this.roleFilter?.value || '';

        this.heroCards.forEach(card => {
            const heroName = card.dataset.heroName.toLowerCase();
            const heroRole = card.dataset.role;
            const matchesSearch = heroName.includes(searchTerm);
            const matchesRole = !selectedRole || heroRole === selectedRole;
            
            card.style.display = matchesSearch && matchesRole ? '' : 'none';
        });
    }

    setupDragAndDrop() {
        this.heroCards.forEach(card => {
            card.setAttribute('draggable', 'true');
            card.addEventListener('dragstart', (e) => this.handleDragStart(e));
        });

        this.draftSlots.forEach(slot => {
            slot.addEventListener('dragover', (e) => this.handleDragOver(e));
            slot.addEventListener('drop', (e) => this.handleDrop(e));
        });
    }

    handleDragStart(e) {
        if (!this.canSelectHero(e.target)) return;
        e.dataTransfer.setData('text/plain', e.target.dataset.heroId);
        this.state.selectedHero = e.target;
    }

    handleDragOver(e) {
        if (this.canFillSlot(e.target)) {
            e.preventDefault();
        }
    }

    handleDrop(e) {
        e.preventDefault();
        const heroId = e.dataTransfer.getData('text/plain');
        const hero = document.querySelector(`[data-hero-id="${heroId}"]`);
        
        if (hero && this.canFillSlot(e.target)) {
            this.fillSlot(e.target, hero);
        }
    }

    setupKeyboardSupport() {
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':  // Space
                    if (this.state.selectedHero && this.state.selectedSlot) {
                        this.lockInSelection();
                        e.preventDefault();
                    }
                    break;
                case 'Escape':
                    this.cancelSelection();
                    e.preventDefault();
                    break;
                case 'b':
                case 'B':
                    this.toggleBanMode();
                    e.preventDefault();
                    break;
            }
        });

        // Make heroes and slots focusable
        this.heroCards.forEach(card => card.setAttribute('tabindex', '0'));
        this.draftSlots.forEach(slot => slot.setAttribute('tabindex', '0'));
    }

    handleHeroClick(e) {
        const hero = e.currentTarget;
        if (!this.canSelectHero(hero)) return;
        
        if (this.state.selectedHero === hero) {
            this.cancelSelection();
        } else {
            this.selectHero(hero);
        }
    }

    handleSlotClick(e) {
        const slot = e.currentTarget;
        if (!this.canFillSlot(slot)) return;

        if (this.state.selectedHero) {
            this.fillSlot(slot, this.state.selectedHero);
        } else {
            this.selectSlot(slot);
        }
    }

    selectHero(hero) {
        if (this.state.selectedHero) {
            this.state.selectedHero.classList.remove('selected');
        }
        
        hero.classList.add('selected');
        this.state.selectedHero = hero;
        
        if (this.state.selectedSlot) {
            this.fillSlot(this.state.selectedSlot, hero);
        }
    }

    selectSlot(slot) {
        if (this.state.selectedSlot) {
            this.state.selectedSlot.classList.remove('selected');
        }
        
        slot.classList.add('selected');
        this.state.selectedSlot = slot;
    }

    fillSlot(slot, hero) {
        const team = slot.dataset.team;
        const index = parseInt(slot.dataset.index);
        const heroData = {
            id: hero.dataset.heroId,
            name: hero.dataset.heroName,
            image: hero.querySelector('img').src
        };

        if (this.state.phase === 'ban') {
            this.state.bans[team][index] = heroData;
        } else {
            this.state.picks[team][index] = heroData;
        }

        // Update UI
        const img = slot.querySelector('img') || document.createElement('img');
        img.src = heroData.image;
        img.alt = heroData.name;
        slot.appendChild(img);
        slot.classList.add('filled');

        // Save state
        this.saveState();
        this.advanceTurn();
    }

    canSelectHero(hero) {
        return !hero.classList.contains('banned') && !hero.classList.contains('picked');
    }

    canFillSlot(slot) {
        const team = slot.dataset.team;
        const index = parseInt(slot.dataset.index);
        const isBanPhase = this.state.phase === 'ban';
        
        return !slot.classList.contains('filled') &&
               ((isBanPhase && index < 3) || (!isBanPhase && index < 5));
    }

    advanceTurn() {
        this.state.turn++;
        
        if (this.state.phase === 'ban' && this.state.turn > 6) {
            this.state.phase = 'pick';
            this.state.turn = 1;
        } else if (this.state.phase === 'pick' && this.state.turn > 10) {
            this.endDraft();
        }

        this.updateUI();
    }

    updateUI() {
        // Update phase display
        this.phaseElement.textContent = `${this.state.phase.toUpperCase()} PHASE - Turn ${this.state.turn}`;
        
        // Reset timer
        this.state.timer = 30;
        this.updateTimer();
        
        // Clear selections
        this.cancelSelection();
        
        // Update available slots
        this.updateSlotStates();
    }

    updateTimer() {
        if (this.timerElement) {
            this.timerElement.textContent = this.state.timer;
        }
    }

    startTimer() {
        setInterval(() => {
            if (this.state.timer > 0) {
                this.state.timer--;
                this.updateTimer();
            }
        }, 1000);
    }

    cancelSelection() {
        if (this.state.selectedHero) {
            this.state.selectedHero.classList.remove('selected');
            this.state.selectedHero = null;
        }
        if (this.state.selectedSlot) {
            this.state.selectedSlot.classList.remove('selected');
            this.state.selectedSlot = null;
        }
    }

    saveState() {
        try {
            Storage.set('draftState', {
                phase: this.state.phase,
                turn: this.state.turn,
                bans: this.state.bans,
                picks: this.state.picks
            });
            
            // Update URL hash for sharing
            window.location.hash = this.state.serialize();
        } catch (e) {
            Toast.show('Failed to save draft state', 'error');
        }
    }

    loadState() {
        try {
            const saved = Storage.get('draftState');
            if (saved) {
                Object.assign(this.state, saved);
                this.updateUI();
            }
        } catch (e) {
            Toast.show('Failed to load saved draft', 'error');
        }
    }

    loadStateFromHash() {
        const hash = window.location.hash.slice(1);
        if (hash) {
            try {
                this.state.deserialize(hash);
                this.updateUI();
                Toast.show('Loaded shared draft', 'success');
            } catch (e) {
                Toast.show('Invalid draft share link', 'error');
            }
        }
    }

    endDraft() {
        Toast.show('Draft completed!', 'success');
        // Additional end-of-draft logic here
    }
}

// Error handling
window.onerror = (msg, url, line, col, error) => {
    console.error('Global error:', { msg, url, line, col, error });
    Toast.show('An error occurred. Please try refreshing the page.', 'error');
    return false;
};

window.onunhandledrejection = (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    Toast.show('An error occurred. Please try refreshing the page.', 'error');
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    const simulator = new DraftSimulator();

    // Load from hash if present
    if (window.location.hash) {
        simulator.loadStateFromHash();
    }
}); 