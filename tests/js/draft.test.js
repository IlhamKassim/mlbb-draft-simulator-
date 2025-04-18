/**
 * @jest-environment jsdom
 */

describe('DraftSimulator', () => {
    let draftSimulator;
    
    beforeEach(() => {
        document.body.innerHTML = `
            <div id="filter-bar">
                <input id="heroSearch">
                <select id="roleFilter">
                    <option value="all">All Roles</option>
                    <option value="mage">Mage</option>
                </select>
                <button id="shareButton">Share Draft</button>
            </div>
            <div class="hero-grid">
                <div class="hero-card" data-hero="Esmeralda" data-role="Mage">
                    <img src="esmeralda.png" class="hero-icon">
                </div>
                <div class="hero-card" data-hero="Lancelot" data-role="Assassin">
                    <img src="lancelot.png" class="hero-icon">
                </div>
            </div>
            <div class="draft-slot" data-team="blue" data-type="ban" data-index="0"></div>
            <div class="draft-slot" data-team="red" data-type="pick" data-index="0"></div>
        `;
        
        // Mock localStorage
        const localStorageMock = {
            getItem: jest.fn(),
            setItem: jest.fn(),
            clear: jest.fn()
        };
        global.localStorage = localStorageMock;
        
        // Import and initialize DraftSimulator
        const DraftSimulator = require('../../static/js/draft.js').DraftSimulator;
        draftSimulator = new DraftSimulator();
    });
    
    test('filterHeroes hides non-matching icons', () => {
        document.getElementById('heroSearch').value = 'esme';
        draftSimulator.filterHeroes();
        
        const esmeraldaCard = document.querySelector('[data-hero="Esmeralda"]');
        const lancelotCard = document.querySelector('[data-hero="Lancelot"]');
        
        expect(esmeraldaCard.style.display).toBe('');
        expect(lancelotCard.style.display).toBe('none');
    });
    
    test('filterHeroes by role', () => {
        document.getElementById('roleFilter').value = 'mage';
        draftSimulator.filterHeroes();
        
        const esmeraldaCard = document.querySelector('[data-hero="Esmeralda"]');
        const lancelotCard = document.querySelector('[data-hero="Lancelot"]');
        
        expect(esmeraldaCard.style.display).toBe('');
        expect(lancelotCard.style.display).toBe('none');
    });
    
    test('saveDraft and restoreDraft roundtrip', () => {
        const state = {
            blueBans: ['Esmeralda'],
            redPicks: ['Lancelot'],
            currentPhase: 'pick',
            currentTeam: 'blue',
            currentIndex: 0
        };
        
        // Save state
        draftSimulator.state = state;
        draftSimulator.saveState();
        
        expect(localStorage.setItem).toHaveBeenCalledWith(
            'draftState',
            JSON.stringify(state)
        );
        
        // Mock localStorage.getItem to return our state
        localStorage.getItem.mockReturnValue(JSON.stringify(state));
        
        // Create new instance to test restore
        const newSimulator = new DraftSimulator();
        expect(newSimulator.state).toEqual(state);
    });
    
    test('shareDraft generates correct URL', () => {
        const state = {
            blueBans: ['Esmeralda'],
            redPicks: ['Lancelot']
        };
        
        draftSimulator.state = state;
        
        // Mock document.execCommand
        document.execCommand = jest.fn();
        
        draftSimulator.shareDraft();
        
        const expectedUrl = `${location.origin}${location.pathname}#state=${encodeURIComponent(JSON.stringify(state))}`;
        expect(document.querySelector('input').value).toBe(expectedUrl);
        expect(document.execCommand).toHaveBeenCalledWith('copy');
    });
    
    test('isValidDrop validates correct slots', () => {
        draftSimulator.state.currentTeam = 'blue';
        draftSimulator.state.currentPhase = 'ban';
        draftSimulator.state.currentIndex = 0;
        
        const validSlot = document.querySelector('[data-team="blue"][data-type="ban"][data-index="0"]');
        const invalidSlot = document.querySelector('[data-team="red"][data-type="pick"][data-index="0"]');
        
        expect(draftSimulator.isValidDrop(validSlot)).toBe(true);
        expect(draftSimulator.isValidDrop(invalidSlot)).toBe(false);
    });
}); 