<?php
// 获取当前页面的文件名
$current_page = basename($_SERVER['PHP_SELF']);
?>
<!-- 侧边栏 -->
<div class="sidebar">
    <div class="logo-container">
        <a href="dashboard.php" class="logo-link">
            <img src="icon.png" alt="FRAS Logo" class="logo">
            <div class="logo-text">FRAS System</div>
        </a>
    </div>

    <ul class="nav-menu">
        <li class="nav-item">
            <a href="dashboard.php" class="nav-link <?php echo ($current_page == 'dashboard.php' || $current_page == 'edit.php' || $current_page == 'addition.php' || $current_page == 'search.php') ? 'active' : ''; ?>">
                <i class="fas fa-tachometer-alt"></i>
                <span><?php echo __('dashboard'); ?></span>
            </a>
        </li>
        <li class="nav-item">
            <a href="statistics.php" class="nav-link <?php echo ($current_page == 'statistics.php') ? 'active' : ''; ?>">
                <i class="fas fa-chart-pie"></i>
                <span><?php echo __('statistics'); ?></span>
            </a>
        </li>
        <li class="nav-item">
            <a href="image_acquisition.php" class="nav-link <?php echo ($current_page == 'image_acquisition.php') ? 'active' : ''; ?>">
                <i class="fas fa-camera"></i>
                <span><?php echo __('image_acquisition'); ?></span>
            </a>
        </li>
        <?php if (isset($_SESSION['role']) && $_SESSION['role'] === 'admin'): ?>
        <li class="nav-item">
            <a href="system_management.php" class="nav-link <?php echo ($current_page == 'system_management.php') ? 'active' : ''; ?>">
                <i class="fas fa-server"></i>
                <span><?php echo __('system_management'); ?></span>
            </a>
        </li>
        <li class="nav-item">
            <a href="monitoring.php" class="nav-link <?php echo ($current_page == 'monitoring.php') ? 'active' : ''; ?>">
                <i class="fas fa-video"></i>
                <span><?php echo __('monitoring'); ?></span>
            </a>
        </li>
        <li class="nav-item">
            <a href="log.php" class="nav-link <?php echo ($current_page == 'log.php') ? 'active' : ''; ?>">
                <i class="fas fa-clipboard-list"></i>
                <span><?php echo __('system_logs'); ?></span>
            </a>
        </li>
        <?php endif; ?>
        <li class="nav-item">
            <a href="settings.php" class="nav-link <?php echo ($current_page == 'settings.php') ? 'active' : ''; ?>">
                <i class="fas fa-cog"></i>
                <span><?php echo __('settings'); ?></span>
            </a>
        </li>
        <li class="nav-item">
            <a href="logout.php" class="nav-link">
                <i class="fas fa-sign-out-alt"></i>
                <span><?php echo __('logout'); ?></span>
            </a>
        </li>
    </ul>
</div>
