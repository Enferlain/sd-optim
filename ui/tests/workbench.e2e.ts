import { expect, test } from '@playwright/test';

test('renders the graph workbench with dependency edges', async ({ page }) => {
	await page.goto('/');

	await expect(page.getByRole('heading', { name: 'Authoring Surface' })).toBeVisible();
	await expect(page.locator('.svelte-flow__node')).toHaveCount(6);
	await expect(page.locator('.svelte-flow__edge')).toHaveCount(6);
	await expect(page.getByRole('heading', { name: 'Checkpoint Sources' })).toBeVisible();
});
