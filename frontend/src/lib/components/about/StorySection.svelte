<script lang="ts">
  import { storyContent } from './aboutContent';
</script>

<section class="section">
  <h2 class="section-title">{storyContent.title}</h2>
  <div class="story-content">
    {#each storyContent.paragraphs as paragraph}
      <p class="section-text">{paragraph}</p>
    {/each}
  </div>
</section>
