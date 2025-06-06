// 主题加载器 - 在所有页面加载时应用保存的主题

// 当DOM加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 尝试从PHP会话中获取主题（如果可用）
    let theme = '';
    
    // 如果页面上有PHP输出的主题设置
    if (typeof phpTheme !== 'undefined' && phpTheme) {
        theme = phpTheme;
    } else {
        // 否则从本地存储获取
        theme = localStorage.getItem('theme');
    }
    
    // 如果没有找到主题设置，使用默认主题
    if (!theme) {
        theme = 'raspberry_pi';
    }
    
    // 应用主题
    applyTheme(theme);
});

// 应用主题函数
function applyTheme(theme) {
    // 移除所有主题属性
    document.documentElement.removeAttribute('data-theme');
    
    // 根据主题名称设置属性
    switch(theme) {
        case 'dark':
            document.documentElement.setAttribute('data-theme', 'dark');
            break;
        case 'fancy':
            document.documentElement.setAttribute('data-theme', 'fancy');
            break;
        case 'ocean':
            document.documentElement.setAttribute('data-theme', 'ocean');
            break;
        case 'raspberry_pi':
        default:
            // 默认主题不需要设置属性
            break;
    }
    
    // 保存到本地存储
    localStorage.setItem('theme', theme);
}
