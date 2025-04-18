class DraftSimulator {
    constructor(options = {}) {
        this.state = this.loadSavedState() || {
            phase: 'ban1',
            currentTeam: 'blue',
            currentTurn: 0,
            timer: null,
            selectedHero: null,
            blueTeam: {
                name: options.blueTeamName || 'Blue Team',
                logo: options.blueTeamLogo || '/static/img/teams/blue_team.png',
                bans: [],
                picks: []
            },
            redTeam: {
                name: options.redTeamName || 'Red Team',
                logo: options.redTeamLogo || '/static/img/teams/red_team.png',
                bans: [],
                picks: []
            },
            bannedHeroes: new Set(),
            selectedHeroes: new Set(),
            timeLeft: 30,
            isLoading: true
        };

        this.timers = {
            ban: 30,
            pick: 45
        };

        this.draftOrder = [
            { phase: 'ban1', team: 'blue' },
            { phase: 'ban1', team: 'red' },
            { phase: 'ban1', team: 'blue' },
            { phase: 'ban1', team: 'red' },
            { phase: 'pick1', team: 'blue' },
            { phase: 'pick1', team: 'red' },
            { phase: 'pick1', team: 'red' },
            { phase: 'pick1', team: 'blue' },
            { phase: 'pick1', team: 'blue' },
            { phase: 'pick1', team: 'red' },
            { phase: 'ban2', team: 'red' },
            { phase: 'ban2', team: 'blue' },
            { phase: 'ban2', team: 'red' },
            { phase: 'ban2', team: 'blue' },
            { phase: 'pick2', team: 'red' },
            { phase: 'pick2', team: 'blue' },
            { phase: 'pick2', team: 'blue' },
            { phase: 'pick2', team: 'red' },
            { phase: 'pick2', team: 'red' },
            { phase: 'pick2', team: 'blue' }
        ];

        this.init();
    }

    async init() {
        try {
            this.showLoadingState();
            
            // Check internet connection
            if (!navigator.onLine) {
                throw new Error('No internet connection. Please check your connection and try again.');
            }

            // Check if heroes data exists in cache
            const cachedHeroes = this.loadHeroesFromCache();
            if (cachedHeroes) {
                this.heroes = cachedHeroes;
                await this.renderHeroGrid();
            } else {
                await this.loadHeroes();
            }

            // Preload hero images
            await this.preloadHeroImages();
            
            this.setupEventListeners();
            this.renderDraftBoard();
            this.startTimer();
            this.updateProgressBar();
            this.hideLoadingState();

            // Setup auto-save
            window.addEventListener('beforeunload', () => this.saveState());
            setInterval(() => this.saveState(), 5000); // Auto-save every 5 seconds
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError(error.message || 'Failed to initialize draft simulator.');
            this.hideLoadingState();
            this.tryLoadingOfflineMode();
        }
    }

    async preloadHeroImages() {
        const imagePromises = Object.keys(this.heroes).map(heroId => {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = resolve;
                img.onerror = () => {
                    console.warn(`Failed to load image for hero: ${heroId}`);
                    resolve(); // Don't reject, just log warning
                };
                img.src = `/static/img/heroes/${heroId.toLowerCase()}.png`;
            });
        });
        await Promise.all(imagePromises);
    }

    saveState() {
        try {
            const stateToSave = {
                ...this.state,
                bannedHeroes: Array.from(this.state.bannedHeroes),
                selectedHeroes: Array.from(this.state.selectedHeroes),
                timer: null // Don't save timer reference
            };
            localStorage.setItem('draftSimulatorState', JSON.stringify(stateToSave));
            localStorage.setItem('draftSimulatorLastSave', new Date().toISOString());
        } catch (error) {
            console.warn('Failed to save state:', error);
        }
    }

    loadSavedState() {
        try {
            const savedState = localStorage.getItem('draftSimulatorState');
            if (!savedState) return null;

            const state = JSON.parse(savedState);
            // Convert arrays back to Sets
            state.bannedHeroes = new Set(state.bannedHeroes);
            state.selectedHeroes = new Set(state.selectedHeroes);
            
            // Check if saved state is too old (more than 1 hour)
            const lastSave = new Date(localStorage.getItem('draftSimulatorLastSave'));
            if (Date.now() - lastSave.getTime() > 3600000) {
                this.clearSavedState();
                return null;
            }

            return state;
        } catch (error) {
            console.warn('Failed to load saved state:', error);
            this.clearSavedState();
            return null;
        }
    }

    clearSavedState() {
        localStorage.removeItem('draftSimulatorState');
        localStorage.removeItem('draftSimulatorLastSave');
    }

    async loadHeroes() {
        try {
            let response = await fetch('/draft/heroes');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (!data || !Array.isArray(data) || data.length === 0) {
                throw new Error('Heroes data is empty or invalid');
            }
            this.heroes = data;
            this.cacheHeroes(data);
            return this.renderHeroGrid();
        } catch (error) {
            throw new Error(`Failed to load heroes data: ${error.message}`);
        }
    }

    cacheHeroes(heroes) {
        try {
            localStorage.setItem('cachedHeroes', JSON.stringify(heroes));
            localStorage.setItem('heroesLastCached', new Date().toISOString());
        } catch (error) {
            console.warn('Failed to cache heroes:', error);
        }
    }

    loadHeroesFromCache() {
        try {
            const cachedHeroes = localStorage.getItem('cachedHeroes');
            const lastCached = localStorage.getItem('heroesLastCached');
            
            if (!cachedHeroes || !lastCached) return null;

            // Check if cache is too old (more than 24 hours)
            if (Date.now() - new Date(lastCached).getTime() > 86400000) {
                localStorage.removeItem('cachedHeroes');
                localStorage.removeItem('heroesLastCached');
                return null;
            }

            return JSON.parse(cachedHeroes);
        } catch (error) {
            console.warn('Failed to load heroes from cache:', error);
            return null;
        }
    }

    async tryLoadingOfflineMode() {
        const cachedHeroes = this.loadHeroesFromCache();
        if (cachedHeroes) {
            this.heroes = cachedHeroes;
            await this.renderHeroGrid();
            this.showToast('Offline Mode', 'Using cached data. Some features may be limited.');
            return true;
        }
        return false;
    }

    setupEventListeners() {
        // Hero grid event listeners
        document.querySelectorAll('.hero-card').forEach(card => {
            card.addEventListener('dragstart', this.handleDragStart.bind(this));
            card.addEventListener('dragend', this.handleDragEnd.bind(this));
            card.addEventListener('click', this.handleHeroClick.bind(this));
            card.setAttribute('tabindex', '0');
            card.addEventListener('keydown', this.handleHeroKeydown.bind(this));
        });

        // Draft slots event listeners
        document.querySelectorAll('.draft-slot').forEach(slot => {
            slot.addEventListener('dragover', this.handleDragOver.bind(this));
            slot.addEventListener('drop', this.handleDrop.bind(this));
            slot.addEventListener('click', this.handleSlotClick.bind(this));
            slot.setAttribute('tabindex', '0');
            slot.addEventListener('keydown', this.handleSlotKeydown.bind(this));
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.cancelSelection();
            } else if (e.key === ' ') {
                e.preventDefault();
                this.lockInSelection();
            } else if (e.ctrlKey && e.key >= '1' && e.key <= '5') {
                e.preventDefault();
                this.quickPick(parseInt(e.key) - 1);
            }
        });
    }

    handleDragStart(e) {
        if (!this.isValidSelection(e.target.dataset.hero)) {
            e.preventDefault();
            return;
        }
        e.dataTransfer.setData('text/plain', e.target.dataset.hero);
        e.target.classList.add('dragging');
    }

    handleDragEnd(e) {
        e.target.classList.remove('dragging');
    }

    handleDragOver(e) {
        if (!this.isValidSlot(e.target)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }

    handleDrop(e) {
        e.preventDefault();
        const hero = e.dataTransfer.getData('text/plain');
        const slot = e.target.closest('.draft-slot');
        
        if (this.isValidSelection(hero) && this.isValidSlot(slot)) {
            this.makeSelection(hero, slot.dataset.position);
        }
    }

    handleHeroClick(e) {
        const heroCard = e.target.closest('.hero-card');
        const hero = heroCard.dataset.hero;
        
        if (this.isValidSelection(hero)) {
            if (this.state.selectedHero === hero) {
                this.cancelSelection();
            } else {
                this.selectHero(hero);
            }
        }
    }

    handleSlotClick(e) {
        const slot = e.target.closest('.draft-slot');
        
        if (this.state.selectedHero && this.isValidSlot(slot)) {
            this.makeSelection(this.state.selectedHero, slot.dataset.position);
        }
    }

    selectHero(hero) {
        this.cancelSelection();
        this.state.selectedHero = hero;
        document.querySelector(`.hero-card[data-hero="${hero}"]`).classList.add('selected');
        this.highlightValidSlots();
    }

    cancelSelection() {
        if (this.state.selectedHero) {
            document.querySelector(`.hero-card[data-hero="${this.state.selectedHero}"]`)?.classList.remove('selected');
            document.querySelectorAll('.draft-slot').forEach(slot => slot.classList.remove('valid-slot'));
            this.state.selectedHero = null;
        }
    }

    lockInSelection() {
        if (this.state.selectedHero) {
            const validSlot = document.querySelector('.draft-slot.valid-slot');
            if (validSlot) {
                this.makeSelection(this.state.selectedHero, validSlot.dataset.position);
            }
        }
    }

    quickPick(index) {
        const currentPhase = this.draftOrder[this.state.currentTurn].phase;
        const currentTeam = this.state.currentTeam;
        const slots = document.querySelectorAll(`.draft-slot[data-phase="${currentPhase}"][data-team="${currentTeam}"]`);
        
        if (slots[index] && this.state.selectedHero) {
            this.makeSelection(this.state.selectedHero, slots[index].dataset.position);
        }
    }

    makeSelection(hero, position) {
        const currentTeam = this.state.currentTeam;
        const currentPhase = this.draftOrder[this.state.currentTurn].phase;
        
        if (currentPhase.startsWith('ban')) {
            this.state.bannedHeroes.add(hero);
            this.state[`${currentTeam}Team`].bans.push(hero);
            this.showBanFeedback(hero);
        } else {
            this.state.selectedHeroes.add(hero);
            this.state[`${currentTeam}Team`].picks.push(hero);
        }

        this.nextTurn();
        this.updateProgressBar();
        this.showSelectionAnimation(hero, currentTeam, currentPhase);
    }

    showBanFeedback(hero) {
        const heroCard = document.querySelector(`.hero-card[data-hero="${hero}"]`);
        heroCard.classList.add('banned');
        const banIcon = document.createElement('div');
        banIcon.className = 'ban-icon';
        heroCard.appendChild(banIcon);
    }

    showSelectionAnimation(hero, team, phase) {
        const slot = document.querySelector(`.draft-slot[data-team="${team}"][data-phase="${phase}"]:empty`);
        if (slot) {
            const img = document.createElement('img');
            img.src = `/static/img/heroes/${hero.toLowerCase()}.png`;
            img.alt = hero;
            img.className = 'hero-portrait animate__animated animate__fadeInDown';
            slot.appendChild(img);
        }
    }

    startTimer() {
        if (this.state.timer) clearInterval(this.state.timer);
        
        this.state.timeLeft = this.getCurrentPhaseTime();
        this.updateTimerDisplay();
        
        this.state.timer = setInterval(() => {
            this.state.timeLeft--;
            this.updateTimerDisplay();
            
            if (this.state.timeLeft <= 0) {
                this.autoSelect();
                clearInterval(this.state.timer);
            }
        }, 1000);
    }

    updateTimerDisplay() {
        const circle = document.querySelector('.timer-progress');
        const text = document.querySelector('.timer-text');
        const totalTime = this.getCurrentPhaseTime();
        const progress = (this.state.timeLeft / totalTime) * 100;
        
        circle.style.strokeDashoffset = `${440 - (440 * progress / 100)}px`;
        text.textContent = this.state.timeLeft;
        
        if (this.state.timeLeft <= 5) {
            circle.classList.add('warning');
            text.classList.add('warning');
        } else {
            circle.classList.remove('warning');
            text.classList.remove('warning');
        }
    }

    updateProgressBar() {
        const totalSteps = this.draftOrder.length;
        const currentStep = this.state.currentTurn;
        const progress = (currentStep / totalSteps) * 100;
        
        document.querySelector('.blue-progress').style.width = `${progress}%`;
        document.querySelector('.red-progress').style.width = `${100 - progress}%`;
    }

    showToast(title, message) {
        const toastContainer = document.querySelector('.toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `
            <div class="toast-header">
                <strong class="me-auto">${title}</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">${message}</div>
        `;
        toastContainer.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }

    nextTurn() {
        this.cancelSelection();
        this.state.currentTurn++;
        
        if (this.state.currentTurn >= this.draftOrder.length) {
            this.endDraft();
            return;
        }

        this.state.currentTeam = this.draftOrder[this.state.currentTurn].team;
        this.state.phase = this.draftOrder[this.state.currentTurn].phase;
        this.startTimer();
        this.updatePhaseIndicator();
    }

    updatePhaseIndicator() {
        const phaseText = document.querySelector('.current-phase');
        const turnText = document.querySelector('.current-turn');
        
        phaseText.textContent = this.state.phase.toUpperCase().replace(/\d+/, ' PHASE');
        turnText.textContent = `${this.state.currentTeam === 'blue' ? 'Your' : 'Enemy'} Turn`;
    }

    endDraft() {
        clearInterval(this.state.timer);
        this.showToast('Draft Complete', 'The draft phase has ended!');
    }

    showLoadingState() {
        const gridContainer = document.querySelector('.hero-grid-container');
        if (!gridContainer) return;
        
        gridContainer.innerHTML = `
            <div class="loading-state">
                <div class="spinner-border text-gold" role="status">
                    <span class="visually-hidden">Loading Heroes...</span>
                </div>
                <p class="mt-2 text-gold">Loading Heroes...</p>
            </div>
        `;
    }

    hideLoadingState() {
        this.state.isLoading = false;
        const loadingState = document.querySelector('.loading-state');
        if (loadingState) {
            loadingState.remove();
        }
    }

    async renderHeroGrid() {
        const grid = document.querySelector('.hero-grid');
        if (!grid) {
            throw new Error('Hero grid element not found');
        }
        
        grid.innerHTML = ''; // Clear existing content

        // Group heroes by role for better organization
        const roleOrder = ['Tank', 'Fighter', 'Assassin', 'Mage', 'Marksman', 'Support'];
        const heroesByRole = {};
        roleOrder.forEach(role => heroesByRole[role] = []);
        
        this.heroes.forEach(hero => {
            if (heroesByRole[hero.role]) {
                heroesByRole[hero.role].push(hero);
            }
        });

        // Create role sections
        roleOrder.forEach(role => {
            if (heroesByRole[role].length > 0) {
                const roleSection = document.createElement('div');
                roleSection.className = 'role-section';
                
                const roleHeader = document.createElement('div');
                roleHeader.className = 'role-header';
                roleHeader.textContent = role;
                roleSection.appendChild(roleHeader);

                const heroesContainer = document.createElement('div');
                heroesContainer.className = 'heroes-container';

                heroesByRole[role].sort((a, b) => a.name.localeCompare(b.name)).forEach(hero => {
                    const heroCard = document.createElement('div');
                    heroCard.className = 'hero-card';
                    heroCard.dataset.hero = hero.name;
                    heroCard.dataset.role = hero.role;
                    heroCard.draggable = true;

                    heroCard.innerHTML = `
                        <img src="/static/img/heroes/${hero.name.toLowerCase().replace(/[^a-z0-9]/g, '')}.png" 
                             alt="${hero.name}" 
                             class="hero-img"
                             loading="lazy">
                        <div class="hero-tooltip">
                            <div class="hero-name">${hero.name}</div>
                            <span class="role-badge ${role.toLowerCase()}">${role}</span>
                        </div>
                    `;

                    // Add event listeners
                    heroCard.addEventListener('dragstart', this.handleDragStart.bind(this));
                    heroCard.addEventListener('dragend', this.handleDragEnd.bind(this));
                    heroCard.addEventListener('click', this.handleHeroClick.bind(this));
                    heroCard.setAttribute('tabindex', '0');
                    heroCard.addEventListener('keydown', this.handleHeroKeydown.bind(this));

                    heroesContainer.appendChild(heroCard);
                });

                roleSection.appendChild(heroesContainer);
                grid.appendChild(roleSection);
            }
        });
    }

    showError(message) {
        // Remove any existing error messages
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message alert alert-danger d-flex align-items-center position-fixed bottom-0 start-50 translate-middle-x mb-4';
        errorDiv.style.zIndex = '1050';
        errorDiv.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-exclamation-triangle-fill flex-shrink-0 me-2" viewBox="0 0 16 16">
                <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
            </svg>
            <div>
                ${message}
            </div>
            <button type="button" class="btn-close ms-3" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        document.body.appendChild(errorDiv);

        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Initialize the simulator when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const simulator = new DraftSimulator();
    
    // Setup online/offline event handlers
    window.addEventListener('online', () => {
        simulator.showToast('Connection Restored', 'You are back online.');
        simulator.init(); // Reinitialize when connection is restored
    });
    
    window.addEventListener('offline', () => {
        simulator.showToast('Connection Lost', 'Working in offline mode. Some features may be limited.');
    });
    
    simulator.init().catch(error => {
        console.error('Failed to initialize draft simulator:', error);
        const toast = new bootstrap.Toast(document.getElementById('errorToast'));
        document.getElementById('errorMessage').textContent = error.message || 'Failed to initialize draft simulator.';
        toast.show();
    });
}); 