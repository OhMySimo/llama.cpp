<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelCapabilities from '../ModelCapabilities.svelte';
	import ModelContext from '../ModelContext.svelte';
	import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
	import ModelId from '../ModelId.svelte';
	import ModelsManagerStatusCell from './ModelsManagerStatusCell.svelte';
	import { modelRowActions, type ModelRowDraftTarget } from './row-actions';
	import {
		canLoadOption,
		configuredContext,
		downloadProgressFor,
		modelDraftBadges,
		type ModelOverride
	} from './utils';
	import { MoreHorizontal } from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import { MODEL_ID, MODEL_ROW_GRID_CLASS, ModelRowDownloadState } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { modelsStore, settingsStore, uiStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		/** Model the pane has open, when this row can be set as its draft. */
		draftTarget?: ModelRowDraftTarget | null;
		isFavorite: (option: ModelOption) => boolean;
		/** Stored per-model overrides, for the drafts and context the row reports. */
		overrides?: Record<string, ModelOverride>;
		option: ModelOption;
		onDelete: (option: ModelOption) => void;
		onSelect: (option: ModelOption) => void;
		onUseAsDraft?: (draft: ModelOption, targetId: string) => void;
		selected: boolean;
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let {
		draftTarget = null,
		indent = 0,
		isFavorite,
		onDelete,
		onSelect,
		onUseAsDraft,
		option,
		overrides,
		selected
	}: Props = $props();

	let favorite = $derived(isFavorite(option));
	let isHidden = $derived(modelsStore.isHidden(option.id));
	// live while the download runs, frozen while it is paused
	let downloadProgress = $derived(downloadProgressFor(option.model));
	// a tracked download takes over the status column while it runs
	let download = $derived(
		modelsStore.status.isDownloadPaused(option.model)
			? ModelRowDownloadState.PAUSED
			: modelsStore.status.isDownloadInProgress(option.model)
				? ModelRowDownloadState.DOWNLOADING
				: null
	);
	let draftBadges = $derived(
		modelDraftBadges(option, overrides?.[option.id]?.load?.speculativeDecoding)
	);

	/** Repo of the row's model, which is what the Discover details are keyed by. */
	function openInDiscover(): void {
		const repo = option.model.split(MODEL_ID.QUANTIZATION_SEPARATOR)[0] ?? option.model;

		uiStore.openModelsDiscover(repo);
	}

	// a tracked download has nothing to configure yet, so its row opens the Discover
	// details, where the sizes, variants and download options live
	function activate(): void {
		if (download) openInDiscover();
		else onSelect(option);
	}

	function handleKeydown(event: KeyboardEvent): void {
		if (event.key === KeyboardKey.SPACE) event.preventDefault();

		if (event.key === KeyboardKey.ENTER || event.key === KeyboardKey.SPACE) activate();
	}
</script>

<div
	class={[
		MODEL_ROW_GRID_CLASS,
		'group relative cursor-pointer rounded-md px-2 py-3 transition',
		isHidden && 'opacity-60',
		selected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
	]}
	onclick={activate}
	onkeydown={handleKeydown}
	role="button"
	tabindex="0"
>
	<span class="flex min-w-0 items-center gap-3" style="padding-left: {indent}px">
		<ModelAvatar
			{option}
			showBaseModelAvatar={!settingsStore.config.groupModelsByFamily}
			showRepoOrgAvatar={settingsStore.config.groupModelsByFamily}
			size="size-9"
		/>

		<span class="flex min-w-0 items-center gap-1.25">
			<ModelId
				aliases={option.aliases}
				class="min-w-0 flex-1"
				draftSidecars={draftBadges}
				hideCapabilities
				hideModalities
				modalities={option.modalities}
				modelId={option.model}
				tags={option.tags}
				title={option.model}
			/>

			<ModelCapabilities {option} />
		</span>
	</span>

	<ModelContext
		class="justify-self-end"
		configured={configuredContext(option, overrides)}
		{option}
	/>

	<ModelsManagerStatusCell {download} {option} />

	{#if download}
		<ModelDownloadProgressBar
			downloadedBytes={downloadProgress?.downloadedBytes ?? 0}
			overlay
			totalBytes={downloadProgress?.totalBytes ?? 0}
		/>
	{/if}

	<div class="flex items-center justify-center justify-self-center">
		<DropdownMenuActions
			actions={modelRowActions(
				option,
				{ canLoad: canLoadOption(option), download, draftTarget, favorite, isHidden },
				{ onDelete, onUseAsDraft }
			)}
			align="end"
			triggerIcon={MoreHorizontal}
			triggerTooltip="Model actions"
		/>
	</div>
</div>
