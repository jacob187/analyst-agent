<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import StorySection from './StorySection.svelte';
  import ArchitectureSection from './ArchitectureSection.svelte';
  import ToolsSection from './ToolsSection.svelte';
  import DashboardSection from './DashboardSection.svelte';
  import './about.css';

  const dispatch = createEventDispatcher<{ back: void }>();
</script>

<div class="about-page">
  <button class="back-btn" on:click={() => dispatch('back')}>
    ← Back to Terminal
  </button>

  <StorySection />
  <ArchitectureSection />
  <ToolsSection />
  <DashboardSection />
</div>
